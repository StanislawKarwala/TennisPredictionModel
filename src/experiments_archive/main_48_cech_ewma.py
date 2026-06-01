"""
Model predykcji wyników meczów tenisowych (ATP) — wersja EWMA
==============================================================
Bazuje na main.py z następującymi zmianami inspirowanymi pracą:
  Q. Dryja, "Data-Driven Prediction of ATP Tennis Match Outcomes",
  VU Amsterdam, BSc Thesis, 2023

Zmiany względem main.py:
  1. Zastąpienie prostych średnich kroczących (SMA) Wykładniczo Ważonymi
     Średnimi Kroczącymi (EWMA):
       EWMA_t = α × m_t + (1 − α) × EWMA_{t-1}
     Dwa horyzonty:
       α = 0.18 (krótkoterminowe, half-life ≈ 3.5 meczu)
       α = 0.05 (długoterminowe, half-life ≈ 13.5 meczu)

  2. Dodanie nowych cech złożonych (composed features):
       - momentum:        zagregowany sygnał trendu (short vs long EWMA)
       - serve_advantage:  siła serwisu − zdolność returnu przeciwnika
       - completeness:     serve_pts_won × return_pts_won

  3. Przyrostowa aktualizacja stanu graczy — O(n) zamiast O(n²)

  4. H2H przechowywane w słowniku — O(1) zamiast skanowania historii

Cechy modelu (48):
    - Kontekst meczu: nawierzchnia, poziom turnieju, best_of (3/5 setów), runda
    - Statyczne gracza: ranking ATP (log), punkty rankingowe (log), wiek, wzrost, ręczność
    - EWMA krótkoterminowe: forma, forma nawierzchniowa, 8 statystyk serwisowych i returnowych
  - EWMA długoterminowe: forma ogólna
  - Złożone: momentum, serve_advantage, completeness
  - Bilans H2H
    - Różnicowe: rank_diff, rank_pts_diff, age_diff, ht_diff, form_diff
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Plik lezy w src/experiments_archive/, wiec parents[2] = katalog projektu.
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "sample_data"

RANDOM_STATE = 42

# =============================================================================
# KONFIGURACJA EWMA
# =============================================================================
# Dwa poziomy wygładzania (Dryja, 2023, sekcja 5.3.4):
#   α = 0.18 → half-life ≈ ln(2)/ln(1/0.82) ≈ 3.5 meczu  (reakcja na bieżącą formę)
#   α = 0.05 → half-life ≈ ln(2)/ln(1/0.95) ≈ 13.5 meczu  (stabilna linia bazowa)
# Momentum = różnica short − long → sygnał wzrostu/spadku formy.

ALPHA_SHORT = 0.18
ALPHA_LONG = 0.05

# Statystyki śledzone przez EWMA (per gracz)
EWMA_STATS = [
    'win_rate', 'ace_rate', 'df_rate', 'first_in_pct', 'first_won_pct',
    'second_won_pct', 'bp_save_pct', 'bp_faced_per_game', 'return_pts_won',
    'serve_pts_won_pct'
]

# Dla tych statystyk niższa wartość oznacza lepszą grę, więc trend dodatni
# wymaga odwrócenia znaku różnicy short - long przy liczeniu momentum.
LOWER_IS_BETTER_STATS = {'df_rate', 'bp_faced_per_game'}

# Wartości domyślne (średnie tourowe ATP) — prior dla nowych graczy
EWMA_DEFAULTS = {
    'win_rate': 0.5,
    'ace_rate': 0.08, 'df_rate': 0.03, 'first_in_pct': 0.60,
    'first_won_pct': 0.70, 'second_won_pct': 0.50,
    'bp_save_pct': 0.60, 'bp_faced_per_game': 0.40,
    'return_pts_won': 0.35, 'serve_pts_won_pct': 0.62
}

# Podzbiór statystyk serwisowych eksportowanych jako cechy modelu
SERVE_STAT_NAMES = [
    'ace_rate', 'df_rate', 'first_in_pct', 'first_won_pct',
    'second_won_pct', 'bp_save_pct', 'bp_faced_per_game', 'return_pts_won'
]


# =============================================================================
# ETAP 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# =============================================================================

df = pd.read_csv(DATA_DIR / "atp_matches_2025.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

cols_serve = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
              'w_SvGms', 'w_bpSaved', 'w_bpFaced',
              'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
              'l_SvGms', 'l_bpSaved', 'l_bpFaced']

cols_base = ['surface', 'tourney_level', 'best_of', 'round',
             'winner_rank', 'winner_age', 'winner_ht', 'winner_hand', 'winner_rank_points',
             'loser_rank', 'loser_age', 'loser_ht', 'loser_hand', 'loser_rank_points',
             'winner_name', 'loser_name'] + cols_serve

df_base = df[cols_base].dropna().copy()
df_base['winner_rank_log'] = np.log(df_base['winner_rank'])
df_base['loser_rank_log'] = np.log(df_base['loser_rank'])
df_base['winner_rank_pts_log'] = np.log(df_base['winner_rank_points'])
df_base['loser_rank_pts_log'] = np.log(df_base['loser_rank_points'])
df_base['winner_is_lefty'] = (df_base['winner_hand'] == 'L').astype(int)
df_base['loser_is_lefty'] = (df_base['loser_hand'] == 'L').astype(int)

ROUND_ORDER = {'R128': 1, 'R64': 2, 'R32': 3, 'RR': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'BR': 6, 'F': 7}
df_base['round_encoded'] = df_base['round'].map(ROUND_ORDER).fillna(3)

print(f"Dane główne (2024): {len(df_base)} meczów")


# =============================================================================
# ETAP 2. DANE HISTORYCZNE (ROZGRZEWKA EWMA)
# =============================================================================
# Im więcej historii, tym dokładniejsze stany EWMA na starcie sezonu 2024.

history_files = [DATA_DIR / f"atp_matches_{year}.csv" for year in range(2001, 2025)]
history_parts = []

for filepath in history_files:
    try:
        df_hist = pd.read_csv(filepath)
        df_hist['tourney_date'] = pd.to_datetime(df_hist['tourney_date'], format='%Y%m%d')
        df_hist = df_hist.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
        df_hist_base = df_hist[cols_base].dropna().copy()
        history_parts.append(df_hist_base)
        print(f"Załadowano dane historyczne ({filepath}): {len(df_hist_base)} meczów")
    except FileNotFoundError:
        print(f"UWAGA: Brak pliku '{filepath}' — pomijam.")

if history_parts:
    df_history_base = pd.concat(history_parts, ignore_index=True)
    print(f"Łączna historia: {len(df_history_base)} meczów")
else:
    print("UWAGA: Brak danych historycznych — stany EWMA startują od domyślnych.")
    df_history_base = pd.DataFrame(columns=cols_base)


# =============================================================================
# ETAP 3. KODOWANIE ZMIENNYCH KATEGORYCZNYCH
# =============================================================================

le_surface = LabelEncoder()
le_level = LabelEncoder()

all_surfaces = pd.concat([df_base['surface'], df_history_base['surface']]).unique()
all_levels = pd.concat([df_base['tourney_level'], df_history_base['tourney_level']]).unique()

le_surface.fit(all_surfaces)
le_level.fit(all_levels)

print(f"Nawierzchnie:      {list(le_surface.classes_)}")
print(f"Poziomy turniejów: {list(le_level.classes_)}")

df_base['surface_encoded'] = le_surface.transform(df_base['surface'])
df_base['tourney_level_encoded'] = le_level.transform(df_base['tourney_level'])


# =============================================================================
# ETAP 4. PODZIAŁ CHRONOLOGICZNY (60/20/20)
# =============================================================================

print("\n=== PODZIAŁ DANYCH (chronologiczny 2024) ===")
train_end = int(len(df_base) * 0.60)
val_end = int(len(df_base) * 0.80)

df_train_raw = df_base.iloc[:train_end].reset_index(drop=True)
df_val_raw = df_base.iloc[train_end:val_end].reset_index(drop=True)
df_test_raw = df_base.iloc[val_end:].reset_index(drop=True)

df_train_raw['match_id'] = range(len(df_train_raw))
df_val_raw['match_id'] = range(len(df_val_raw))
df_test_raw['match_id'] = range(len(df_test_raw))

print(f"Trening:    {len(df_train_raw)} meczów")
print(f"Walidacja:  {len(df_val_raw)} meczów")
print(f"Test:       {len(df_test_raw)} meczów")


# =============================================================================
# ETAP 5. CECHY DYNAMICZNE — EWMA (Exponentially Weighted Moving Averages)
# =============================================================================
# Zamiast prostej średniej kroczącej (SMA: mean z okna 10 meczów), stosujemy EWMA:
#
#   EWMA_t = α × m_t + (1 − α) × EWMA_{t-1}
#
# Zalety EWMA wobec SMA:
#   - Nowsze mecze mają wykładniczo większą wagę → szybsza reakcja na zmiany formy
#   - Brak „efektu krawędzi" — wyjście starego meczu z okna nie powoduje skoku
#   - Dwa horyzonty (short/long) umożliwiają wychwycenie trendu (momentum)
#   - Aktualizacja przyrostowa O(1) na mecz zamiast O(n) skanowania okna
#
# Stan EWMA utrzymywany jest per gracz i aktualizowany chronologicznie.
# Dzięki temu przetworzenie danych to jeden przebieg O(n) po wszystkich meczach,
# zamiast O(n²) jak w wersji z SMA.


def init_player_state():
    """Inicjalizuje stan EWMA dla nowego gracza (wartości domyślne = średnie ATP)."""
    return {
        'short': {stat: EWMA_DEFAULTS[stat] for stat in EWMA_STATS},
        'long': {stat: EWMA_DEFAULTS[stat] for stat in EWMA_STATS},
        'surface_short': {},  # nawierzchnia → EWMA win_rate (krótkoterminowe)
        'n_matches': 0
    }


def extract_match_stats(row, is_winner):
    """
    Wyodrębnia statystyki meczu dla gracza (zwycięzca lub przegrany).

    Czyta surowe kolumny w_*/l_* i przelicza na wskaźniki procentowe.
    Zwraca dict ze statystykami, które udało się obliczyć (NaN-safe).
    """
    if is_winner:
        pfx, opp = 'w_', 'l_'
    else:
        pfx, opp = 'l_', 'w_'

    stats = {'win_rate': 1.0 if is_winner else 0.0}

    svpt = row[f'{pfx}svpt']
    first_in = row[f'{pfx}1stIn']

    if svpt > 0:
        stats['ace_rate'] = row[f'{pfx}ace'] / svpt
        stats['df_rate'] = row[f'{pfx}df'] / svpt
        stats['first_in_pct'] = first_in / svpt
        first_won = row[f'{pfx}1stWon']
        second_won = row[f'{pfx}2ndWon']
        stats['serve_pts_won_pct'] = (first_won + second_won) / svpt

        if first_in > 0:
            stats['first_won_pct'] = first_won / first_in
        second_serve = svpt - first_in
        if second_serve > 0:
            stats['second_won_pct'] = second_won / second_serve

    bp_faced = row[f'{pfx}bpFaced']
    if bp_faced > 0:
        stats['bp_save_pct'] = row[f'{pfx}bpSaved'] / bp_faced
    sv_gms = row[f'{pfx}SvGms']
    if sv_gms > 0:
        stats['bp_faced_per_game'] = bp_faced / sv_gms

    opp_svpt = row[f'{opp}svpt']
    if opp_svpt > 0:
        opp_first_won = row[f'{opp}1stWon']
        opp_second_won = row[f'{opp}2ndWon']
        stats['return_pts_won'] = (opp_svpt - opp_first_won - opp_second_won) / opp_svpt

    return stats


def update_player_ewma(state, match_stats, surface):
    """
    Aktualizuje stan EWMA gracza po rozegranym meczu.

    Dla każdej dostępnej statystyki:
      EWMA_nowe = α × obserwacja + (1 − α) × EWMA_stare

    Aktualizuje zarówno krótko- jak i długoterminowe EWMA
    oraz formę na konkretnej nawierzchni.
    """
    for stat in EWMA_STATS:
        if stat in match_stats:
            state['short'][stat] = (ALPHA_SHORT * match_stats[stat]
                                    + (1 - ALPHA_SHORT) * state['short'][stat])
            state['long'][stat] = (ALPHA_LONG * match_stats[stat]
                                   + (1 - ALPHA_LONG) * state['long'][stat])

    # Forma na nawierzchni — krótkoterminowe EWMA
    if surface not in state['surface_short']:
        state['surface_short'][surface] = EWMA_DEFAULTS['win_rate']
    state['surface_short'][surface] = (
        ALPHA_SHORT * match_stats['win_rate']
        + (1 - ALPHA_SHORT) * state['surface_short'][surface]
    )

    state['n_matches'] += 1


def get_player_features(state, surface):
    """
    Odczytuje bieżące cechy gracza z jego stanu EWMA.

    Zwraca dict z:
      - form_short/long: ogólna forma (EWMA win_rate)
      - surface_form: forma na nawierzchni (EWMA krótkoterminowe)
      - 8 statystyk serwisowych (krótkoterminowe EWMA)
      - serve_pts_won_pct: do obliczenia cech złożonych
      - completeness: serve_pts_won × return_pts_won
      - momentum: zagregowany sygnał trendu (short vs long)
    """
    feats = {}

    # Forma ogólna
    feats['form_short'] = state['short']['win_rate']
    feats['form_long'] = state['long']['win_rate']

    # Forma na nawierzchni (fallback na formę ogólną)
    feats['surface_form'] = state['surface_short'].get(surface, state['short']['win_rate'])

    # Statystyki serwisowe — krótkoterminowe EWMA
    for stat in SERVE_STAT_NAMES:
        feats[stat] = state['short'][stat]
    feats['serve_pts_won_pct'] = state['short']['serve_pts_won_pct']

    # Completeness = zdolność serwisowa × zdolność returnowa (Dryja, sekcja 5.3.6)
    feats['completeness'] = state['short']['serve_pts_won_pct'] * state['short']['return_pts_won']

    # Momentum = zagregowany sygnał trendu Short vs Long (Dryja, sekcja 5.3.7)
    # Dla każdej statystyki: sign(EWMA_short − EWMA_long), uśrednione → zakres [-1, +1]
    # Wartość > 0: gracz w trendzie wzrostowym, < 0: w trendzie spadkowym
    momentum_signals = []
    for stat in EWMA_STATS:
        diff = state['short'][stat] - state['long'][stat]
        if stat in LOWER_IS_BETTER_STATS:
            diff = -diff
        if diff > 0:
            momentum_signals.append(1)
        elif diff < 0:
            momentum_signals.append(-1)
        else:
            momentum_signals.append(0)
    feats['momentum'] = sum(momentum_signals) / len(momentum_signals)

    return feats


def get_h2h(h2h_record, p1, p2):
    """Bilans bezpośrednich spotkań: p1_wins − p2_wins. O(1) z dict."""
    return h2h_record.get((p1, p2), 0) - h2h_record.get((p2, p1), 0)


def initialize_ewma_states(historical_data):
    """
    Przetwarza dane historyczne chronologicznie, aby rozgrzać stany EWMA graczy.
    Jednocześnie buduje słownik H2H.

    Parametry:
        historical_data: DataFrame z meczami (posortowany chronologicznie)
    Zwraca:
        (player_states dict, h2h_record dict)
    """
    player_states = {}
    h2h_record = {}

    for _, row in historical_data.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row['surface']

        for player in [winner, loser]:
            if player not in player_states:
                player_states[player] = init_player_state()

        w_stats = extract_match_stats(row, is_winner=True)
        l_stats = extract_match_stats(row, is_winner=False)

        update_player_ewma(player_states[winner], w_stats, surface)
        update_player_ewma(player_states[loser], l_stats, surface)

        key = (winner, loser)
        h2h_record[key] = h2h_record.get(key, 0) + 1

    return player_states, h2h_record


def add_dynamic_features(df_subset, player_states, h2h_record):
    """
    Dołącza cechy dynamiczne EWMA do każdego meczu w df_subset.

    Dla i-tego meczu:
      1. Odczytuje bieżący stan EWMA obu graczy (PRZED meczem)
      2. Oblicza cechy złożone (serve_advantage, momentum, completeness)
      3. Oblicza H2H z historii par
      4. Aktualizuje stany EWMA i H2H (PO meczu)

    UWAGA: Mutuje player_states i h2h_record (aktualizacja przyrostowa).
    Kolejne wywołania (train → val → test) kumulują stan.
    """
    feature_rows = []

    for i in range(len(df_subset)):
        row = df_subset.iloc[i]
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row['surface']

        for player in [winner, loser]:
            if player not in player_states:
                player_states[player] = init_player_state()

        # ODCZYT stanu EWMA (przed tym meczem)
        w_feats = get_player_features(player_states[winner], surface)
        l_feats = get_player_features(player_states[loser], surface)

        # Serve Advantage = siła serwisu gracza − zdolność returnu przeciwnika
        # (Dryja, sekcja 5.3.5)
        w_serve_adv = w_feats['serve_pts_won_pct'] - l_feats['return_pts_won']
        l_serve_adv = l_feats['serve_pts_won_pct'] - w_feats['return_pts_won']

        # H2H — O(1) z dict
        h2h = get_h2h(h2h_record, winner, loser)

        feature_rows.append({
            'h2h_diff': h2h,
            'w_form_short': w_feats['form_short'],
            'l_form_short': l_feats['form_short'],
            'w_form_long': w_feats['form_long'],
            'l_form_long': l_feats['form_long'],
            'w_surface_form': w_feats['surface_form'],
            'l_surface_form': l_feats['surface_form'],
            'w_momentum': w_feats['momentum'],
            'l_momentum': l_feats['momentum'],
            'w_serve_advantage': w_serve_adv,
            'l_serve_advantage': l_serve_adv,
            'w_completeness': w_feats['completeness'],
            'l_completeness': l_feats['completeness'],
            **{f'w_{s}': w_feats[s] for s in SERVE_STAT_NAMES},
            **{f'l_{s}': l_feats[s] for s in SERVE_STAT_NAMES},
        })

        # AKTUALIZACJA stanu EWMA (po tym meczu)
        w_stats = extract_match_stats(row, is_winner=True)
        l_stats = extract_match_stats(row, is_winner=False)
        update_player_ewma(player_states[winner], w_stats, surface)
        update_player_ewma(player_states[loser], l_stats, surface)

        # Aktualizacja H2H
        key = (winner, loser)
        h2h_record[key] = h2h_record.get(key, 0) + 1

    # Dołączenie obliczonych cech do DataFrame
    df_result = df_subset.copy()
    for col in feature_rows[0]:
        df_result[col] = [r[col] for r in feature_rows]

    return df_result


# Inicjalizacja stanów EWMA z danych historycznych (2018–2023)
print("\nInicjalizacja stanów EWMA z danych historycznych...")
player_states, h2h_record = initialize_ewma_states(df_history_base)
print(f"Rozgrzano stany EWMA dla {len(player_states)} graczy")

# Obliczanie cech dynamicznych — stan EWMA przepływa sekwencyjnie: train → val → test
print("Obliczanie cech dynamicznych (EWMA)...")
df_train_raw = add_dynamic_features(df_train_raw, player_states, h2h_record)
df_val_raw = add_dynamic_features(df_val_raw, player_states, h2h_record)
df_test_raw = add_dynamic_features(df_test_raw, player_states, h2h_record)


# =============================================================================
# ETAP 6. SYMETRYZACJA DANYCH
# =============================================================================

def symmetrize_data(df_subset, shuffle=True):
    """
    Tworzy symetryczny (zbalansowany) zbiór danych z zamianą ról graczy.
    Dla każdego meczu: 2 wiersze (perspektywa zwycięzcy i przegranego).
    """
    rows_p1_wins = []
    rows_p2_wins = []

    for idx, row in df_subset.iterrows():
        # Perspektywa 1: Gracz 1 = zwycięzca → y = 1
        row1 = {
            'match_id': row['match_id'],
            'surface': row['surface_encoded'],
            'tourney_level': row['tourney_level_encoded'],
            'best_of': row['best_of'],
            'round_num': row['round_encoded'],
            'p1_rank_log': row['winner_rank_log'],
            'p1_rank_pts_log': row['winner_rank_pts_log'],
            'p1_age': row['winner_age'],
            'p1_ht': row['winner_ht'],
            'p1_is_lefty': row['winner_is_lefty'],
            'p2_rank_log': row['loser_rank_log'],
            'p2_rank_pts_log': row['loser_rank_pts_log'],
            'p2_age': row['loser_age'],
            'p2_ht': row['loser_ht'],
            'p2_is_lefty': row['loser_is_lefty'],
            'p1_h2h': row['h2h_diff'],
            'p1_form_short': row['w_form_short'],
            'p2_form_short': row['l_form_short'],
            'p1_form_long': row['w_form_long'],
            'p2_form_long': row['l_form_long'],
            'p1_surface_form': row['w_surface_form'],
            'p2_surface_form': row['l_surface_form'],
            'p1_momentum': row['w_momentum'],
            'p2_momentum': row['l_momentum'],
            'p1_serve_advantage': row['w_serve_advantage'],
            'p2_serve_advantage': row['l_serve_advantage'],
            'p1_completeness': row['w_completeness'],
            'p2_completeness': row['l_completeness'],
            'rank_diff': row['winner_rank_log'] - row['loser_rank_log'],
            'rank_pts_diff': row['winner_rank_pts_log'] - row['loser_rank_pts_log'],
            'age_diff': row['winner_age'] - row['loser_age'],
            'ht_diff': row['winner_ht'] - row['loser_ht'],
            'form_diff': row['w_form_short'] - row['l_form_short'],
            'y': 1,
            'actual_winner': row['winner_name'],
            'actual_loser': row['loser_name'],
            'p1_name': row['winner_name'],
            'p2_name': row['loser_name']
        }
        for stat in SERVE_STAT_NAMES:
            row1[f'p1_{stat}'] = row[f'w_{stat}']
            row1[f'p2_{stat}'] = row[f'l_{stat}']

        # Perspektywa 2: Gracz 1 = przegrany → y = 0
        row2 = {
            'match_id': row['match_id'],
            'surface': row['surface_encoded'],
            'tourney_level': row['tourney_level_encoded'],
            'best_of': row['best_of'],
            'round_num': row['round_encoded'],
            'p1_rank_log': row['loser_rank_log'],
            'p1_rank_pts_log': row['loser_rank_pts_log'],
            'p1_age': row['loser_age'],
            'p1_ht': row['loser_ht'],
            'p1_is_lefty': row['loser_is_lefty'],
            'p2_rank_log': row['winner_rank_log'],
            'p2_rank_pts_log': row['winner_rank_pts_log'],
            'p2_age': row['winner_age'],
            'p2_ht': row['winner_ht'],
            'p2_is_lefty': row['winner_is_lefty'],
            'p1_h2h': -row['h2h_diff'],
            'p1_form_short': row['l_form_short'],
            'p2_form_short': row['w_form_short'],
            'p1_form_long': row['l_form_long'],
            'p2_form_long': row['w_form_long'],
            'p1_surface_form': row['l_surface_form'],
            'p2_surface_form': row['w_surface_form'],
            'p1_momentum': row['l_momentum'],
            'p2_momentum': row['w_momentum'],
            'p1_serve_advantage': row['l_serve_advantage'],
            'p2_serve_advantage': row['w_serve_advantage'],
            'p1_completeness': row['l_completeness'],
            'p2_completeness': row['w_completeness'],
            'rank_diff': row['loser_rank_log'] - row['winner_rank_log'],
            'rank_pts_diff': row['loser_rank_pts_log'] - row['winner_rank_pts_log'],
            'age_diff': row['loser_age'] - row['winner_age'],
            'ht_diff': row['loser_ht'] - row['winner_ht'],
            'form_diff': row['l_form_short'] - row['w_form_short'],
            'y': 0,
            'actual_winner': row['winner_name'],
            'actual_loser': row['loser_name'],
            'p1_name': row['loser_name'],
            'p2_name': row['winner_name']
        }
        for stat in SERVE_STAT_NAMES:
            row2[f'p1_{stat}'] = row[f'l_{stat}']
            row2[f'p2_{stat}'] = row[f'w_{stat}']

        rows_p1_wins.append(row1)
        rows_p2_wins.append(row2)

    all_rows = []
    for r1, r2 in zip(rows_p1_wins, rows_p2_wins):
        all_rows.extend([r1, r2])

    result = pd.DataFrame(all_rows)

    if shuffle:
        result = result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)

    return result


val_data = symmetrize_data(df_val_raw, shuffle=True)
test_data = symmetrize_data(df_test_raw, shuffle=True)

print(f"\nPo symetryzacji:")
print(f"Walidacja:  {len(val_data)} próbek (y=1: {sum(val_data['y']==1)}, y=0: {sum(val_data['y']==0)})")
print(f"Test:       {len(test_data)} próbek (y=1: {sum(test_data['y']==1)}, y=0: {sum(test_data['y']==0)})")


# =============================================================================
# ETAP 7. DEFINICJA WEKTORA CECH (48 cech)
# =============================================================================

features = [
    'surface', 'tourney_level', 'best_of', 'round_num',
    'p1_rank_log', 'p1_rank_pts_log', 'p1_age', 'p1_ht', 'p1_is_lefty',
    'p2_rank_log', 'p2_rank_pts_log', 'p2_age', 'p2_ht', 'p2_is_lefty',
    'p1_h2h',
    # EWMA forma (krótko- i długoterminowe)
    'p1_form_short', 'p2_form_short',
    'p1_form_long', 'p2_form_long',
    # EWMA forma na nawierzchni
    'p1_surface_form', 'p2_surface_form',
    # Cechy złożone (composed)
    'p1_momentum', 'p2_momentum',
    'p1_serve_advantage', 'p2_serve_advantage',
    'p1_completeness', 'p2_completeness',
    # EWMA statystyki serwisowe (krótkoterminowe)
    'p1_ace_rate', 'p2_ace_rate',
    'p1_df_rate', 'p2_df_rate',
    'p1_first_in_pct', 'p2_first_in_pct',
    'p1_first_won_pct', 'p2_first_won_pct',
    'p1_second_won_pct', 'p2_second_won_pct',
    'p1_bp_save_pct', 'p2_bp_save_pct',
    'p1_bp_faced_per_game', 'p2_bp_faced_per_game',
    'p1_return_pts_won', 'p2_return_pts_won',
    # Cechy różnicowe
    'rank_diff', 'rank_pts_diff', 'age_diff', 'ht_diff', 'form_diff'
]

print(f"\nLiczba cech: {len(features)}")

X_val = val_data[features]
y_val = val_data['y']

X_test = test_data[features]
y_test = test_data['y']


# =============================================================================
# ETAP 8. OPTYMALIZACJA HIPERPARAMETRÓW (RandomizedSearchCV + TimeSeriesSplit)
# =============================================================================

train_data_ordered = symmetrize_data(df_train_raw, shuffle=False)
X_train_cv = train_data_ordered[features]
y_train_cv = train_data_ordered['y']

print(f"Próbki treningowe dla walidacji krzyżowej: {len(train_data_ordered)}")

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'max_samples': [0.7, 0.8, 0.9, 1.0]
}

rf = RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE)
tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    rf,
    param_dist,
    n_iter=50,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)

search.fit(X_train_cv, y_train_cv)

print(f"\nNajlepsze hiperparametry: {search.best_params_}")
print(f"Najlepszy wynik CV (chronologiczny): {search.best_score_:.4f}")

best_rf = search.best_estimator_
best_rf.n_jobs = -1

train_data_final = symmetrize_data(df_train_raw, shuffle=True)
X_train_final = train_data_final[features]
y_train_final = train_data_final['y']

print(f"Trening finalnego modelu na {len(train_data_final)} próbkach...")
best_rf.fit(X_train_final, y_train_final)


# =============================================================================
# ETAP 9. EWALUACJA MODELU
# =============================================================================

val_pred = best_rf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)

print("\n=== WYNIKI NA ZBIORZE WALIDACYJNYM ===")
print(f"Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, val_pred, target_names=['Gracz 2 wygrywa', 'Gracz 1 wygrywa']))
print("\nMacierz pomyłek:")
print(confusion_matrix(y_val, val_pred))

test_pred = best_rf.predict(X_test)
test_pred_proba = best_rf.predict_proba(X_test)
test_acc = accuracy_score(y_test, test_pred)

print("\n" + "="*50)
print("=== FINALNE WYNIKI NA ZBIORZE TESTOWYM ===")
print("="*50)
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=['Gracz 2 wygrywa', 'Gracz 1 wygrywa']))
print("\nMacierz pomyłek:")
print(confusion_matrix(y_test, test_pred))


# =============================================================================
# ETAP 10. PREDYKCJA NA POZIOMIE MECZÓW
# =============================================================================

print("\n" + "="*50)
print("=== PRZEWIDYWANIE ZWYCIĘZCÓW MECZÓW ===")
print("="*50)

test_data['p1_win_probability'] = test_pred_proba[:, 1]

winner_perspective = test_data[test_data['y'] == 1].copy()
winner_perspective['predicted_winner'] = winner_perspective.apply(
    lambda row: row['p1_name'] if row['p1_win_probability'] > 0.5 else row['p2_name'],
    axis=1
)
winner_perspective['correct_prediction'] = winner_perspective['p1_win_probability'] > 0.5
match_accuracy = winner_perspective['correct_prediction'].mean()

print(f"\nACCURACY PRZEWIDYWANIA ZWYCIĘZCÓW: {match_accuracy:.4f} ({match_accuracy*100:.2f}%)")
print(f"Poprawnie przewidziane: {int(winner_perspective['correct_prediction'].sum())} / {len(winner_perspective)} meczów")

print("\nPrzykładowe predykcje (pierwsze 10 meczów):")
print("-" * 80)
sample_predictions = winner_perspective.head(10)[['p1_name', 'p2_name', 'actual_winner',
                                                   'predicted_winner', 'p1_win_probability',
                                                   'correct_prediction']]
for idx, row in sample_predictions.iterrows():
    status = " OK " if row['correct_prediction'] else "MISS"
    print(f"[{status}] {row['p1_name'][:20]:20s} vs {row['p2_name'][:20]:20s}")
    print(f"       Rzeczywisty: {row['actual_winner'][:25]:25s} | Przewidziany: {row['predicted_winner'][:25]:25s}")
    print(f"       P(zwycięzca wygra): {row['p1_win_probability']:.3f}")
    print()


# =============================================================================
# ETAP 11. WAŻNOŚĆ CECH
# =============================================================================

print("\n=== WAŻNOŚĆ CECH ===")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string())


# =============================================================================
# ETAP 12. ANALIZA BŁĘDÓW
# =============================================================================

print("\n" + "="*50)
print("=== ANALIZA NAJWIĘKSZYCH BŁĘDÓW ===")
print("="*50)

winner_perspective['error'] = 1 - winner_perspective['p1_win_probability']

worst_predictions = winner_perspective.nlargest(5, 'error')[['p1_name', 'p2_name', 'actual_winner',
                                                              'predicted_winner', 'p1_win_probability', 'error']]
print("\nTop 5 najgorszych predykcji:")
for idx, row in worst_predictions.iterrows():
    print(f"Mecz: {row['p1_name']} vs {row['p2_name']}")
    print(f"  Rzeczywisty zwycięzca: {row['actual_winner']}")
    print(f"  Przewidziany: {row['predicted_winner']} (P(zwycięzca): {row['p1_win_probability']:.3f})")
    print(f"  Błąd: {row['error']:.3f}\n")


# =============================================================================
# ETAP 13. PODSUMOWANIE I PORÓWNANIE
# =============================================================================

print("\n" + "="*50)
print("=== PODSUMOWANIE (EWMA) ===")
print("="*50)
print(f"Statystyki modelu EWMA:")
print(f"  Liczba cech:         {len(features)}")
print(f"  α_short = {ALPHA_SHORT}, α_long = {ALPHA_LONG}")
print(f"  CV Accuracy:         {search.best_score_:.4f}")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Test Accuracy:       {test_acc:.4f}")
print(f"  Match Prediction:    {match_accuracy:.4f}")
print()
print(f"Porównanie z main.py (SMA, 48 cech):")
print(f"  main.py Match Prediction:  0.6153 (61.53%)")
print(f"  EWMA    Match Prediction:  {match_accuracy:.4f} ({match_accuracy*100:.2f}%)")
print(f"  Różnica: {(match_accuracy - 0.6153)*100:+.2f} p.p.")
print()
print(f"Baseline (losowe): 0.5000 (50%)")
print(f"Mój model EWMA:   {match_accuracy:.4f} ({match_accuracy*100:.1f}%)")
print(f"Przewaga nad losowym zgadywaniem: +{(match_accuracy-0.5)*100:.1f} p.p.")
