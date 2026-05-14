"""
Model predykcji wyników meczów tenisowych (ATP)
================================================
Metodologia:
  - Algorytm: Random Forest Classifier z optymalizacją hiperparametrów (RandomizedSearchCV)
  - Walidacja: TimeSeriesSplit (walidacja krzyżowa z zachowaniem porządku chronologicznego)
  - Podział danych: chronologiczny 60% trening / 20% walidacja / 20% test (rok 2024)
  - Redukcja cold-start: dane z sezonów 2018–2023 jako kontekst historyczny dla cech dynamicznych

Cechy modelu (40):
  - Kontekst meczu: nawierzchnia, poziom turnieju, best_of (3/5 setów), runda
  - Statyczne gracza: ranking ATP (log), punkty rankingowe (log), wiek, wzrost, ręczność
  - Dynamiczne: forma gracza (win rate z ostatnich 10 meczów),
                forma nawierzchniowa, bilans H2H,
                statystyki serwisowe i returnowe (rolling avg z 10 meczów)
  - Pochodne: cechy różnicowe (rank_diff, rank_pts_diff, age_diff, ht_diff, form_diff)
  - Symetryzacja danych: każdy mecz generuje dwa przykłady treningowe
    (zamiana ról Gracz 1 / Gracz 2), co eliminuje pozycyjne obciążenie modelu (positional bias)

Źródło danych: Jeff Sackmann / tennis_atp (format CSV z kolumnami winner_*/loser_*)
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42  # Ziarno losowości 


# =============================================================================
# ETAP 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# =============================================================================
# Dane pochodzą z plików CSV w formacie Jeff Sackmann (tennis_atp).
# Każdy wiersz opisuje jeden mecz z perspektywy zwycięzcy i przegranego.
# Sortowanie chronologiczne jest kluczowe — model nie może „widzieć" przyszłości.

df = pd.read_csv('sample_data/2024.csv')
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

# Transformacja logarytmiczna rankingów ATP.
# Uzasadnienie: rozkład rankingów jest silnie prawoskośny — różnica między
# pozycją 1 a 10 ma większe znaczenie niż między 90 a 100. Logarytm
# kompresuje ogon rozkładu i lepiej oddaje nieliniową relację ranking → siła gracza.
df_base['winner_rank_log'] = np.log(df_base['winner_rank'])
df_base['loser_rank_log'] = np.log(df_base['loser_rank'])

# Logarytm punktów rankingowych — dodatkowa granularność ponad sam ranking.
df_base['winner_rank_pts_log'] = np.log(df_base['winner_rank_points'])
df_base['loser_rank_pts_log'] = np.log(df_base['loser_rank_points'])

# Ręczność gracza — leworęczni mają statystyczną przewagę w niektórych matchupach.
df_base['winner_is_lefty'] = (df_base['winner_hand'] == 'L').astype(int)
df_base['loser_is_lefty'] = (df_base['loser_hand'] == 'L').astype(int)

# Runda turnieju — ordinalnie zakodowana (im dalszy etap, tym wyższa wartość).
ROUND_ORDER = {'R128': 1, 'R64': 2, 'R32': 3, 'RR': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'BR': 6, 'F': 7}
df_base['round_encoded'] = df_base['round'].map(ROUND_ORDER).fillna(3)

print(f"Dane główne (2024): {len(df_base)} meczów")


# =============================================================================
# ETAP 2. DANE HISTORYCZNE (REDUKCJA PROBLEMU COLD START)
# =============================================================================
# Cechy dynamiczne (forma, H2H) wymagają historii meczów danego gracza.
# Na początku sezonu 2024 taka historia nie istnieje (problem „zimnego startu").
# Rozwiązanie: wczytanie danych z sezonów 2018–2023 jako bazy do obliczeń
# cech dynamicznych. Im więcej historii, tym dokładniejsze oszacowania formy
# i bilansu bezpośrednich spotkań (H2H), szczególnie dla rzadkich par graczy.

history_files = ['sample_data/2018.csv', 'sample_data/2019.csv', 'sample_data/2020.csv',
                 'sample_data/2021.csv', 'sample_data/2022.csv', 'sample_data/2023.csv']
history_parts = []

for filepath in history_files:
    try:
        df_hist = pd.read_csv(filepath)
        df_hist['tourney_date'] = pd.to_datetime(df_hist['tourney_date'], format='%Y%m%d')
        df_hist = df_hist.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
        df_hist_base = df_hist[cols_base].dropna().copy()
        history_parts.append(df_hist_base)
        print(f"Zaladowano dane historyczne ({filepath}): {len(df_hist_base)} meczow")
    except FileNotFoundError:
        print(f"UWAGA: Brak pliku '{filepath}' -- pomijam.")

if history_parts:
    df_history_base = pd.concat(history_parts, ignore_index=True)
    print(f"Laczna historia: {len(df_history_base)} meczow")
else:
    print("UWAGA: Brak danych historycznych -- cechy dynamiczne beda niedokladne.")
    df_history_base = pd.DataFrame(columns=cols_base)


# =============================================================================
# ETAP 3. KODOWANIE ZMIENNYCH KATEGORYCZNYCH (Label Encoding)
# =============================================================================
# Nawierzchnia (Hard/Clay/Grass) i poziom turnieju (Grand Slam/Masters/250/...)
# to zmienne kategoryczne. LabelEncoder przypisuje im wartości liczbowe.
# Enkodery dopasowywane są na zbiorze 2018–2023+2024, aby uwzględnić wszystkie
# możliwe kategorie i uniknąć błędów przy transformacji.

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
# ETAP 4. PODZIAŁ CHRONOLOGICZNY (Train / Validation / Test)
# =============================================================================
# Podział danych musi zachowywać porządek czasowy, ponieważ mecze tenisowe tworzą
# szereg czasowy — model powinien być oceniany na danych z przyszłości względem
# danych treningowych. Losowy podział (np. train_test_split z shuffle=True)
# prowadziłby do wycieku informacji z przyszłości (data leakage).
#
# Proporcje: 60% trening — 20% walidacja — 20% test
# Chronologia: trening < walidacja < test (wg daty turnieju)

print("\n=== PODZIAŁ DANYCH (chronologiczny 2024) ===")
train_end = int(len(df_base) * 0.60)
val_end = int(len(df_base) * 0.80)

df_train_raw = df_base.iloc[:train_end].reset_index(drop=True)
df_val_raw = df_base.iloc[train_end:val_end].reset_index(drop=True)
df_test_raw = df_base.iloc[val_end:].reset_index(drop=True)

# Identyfikator meczu umożliwia późniejsze łączenie par symetrycznych
# (ten sam mecz widziany z dwóch perspektyw) z powrotem w jeden wynik.
df_train_raw['match_id'] = range(len(df_train_raw))
df_val_raw['match_id'] = range(len(df_val_raw))
df_test_raw['match_id'] = range(len(df_test_raw))

print(f"Trening:    {len(df_train_raw)} meczów")
print(f"Walidacja:  {len(df_val_raw)} meczów")
print(f"Test:       {len(df_test_raw)} meczów")

# =============================================================================
# ETAP 5. CECHY DYNAMICZNE — Forma gracza i bilans bezpośrednich pojedynków
# =============================================================================
# Cechy statyczne (ranking, wiek, wzrost) nie zmieniają się w trakcie turnieju,
# ale forma sportowa i historia bezpośrednich spotkań (Head-to-Head) ewoluują
# z każdym rozegranym meczem.
#
# WAŻNE: Dla każdego meczu cechy dynamiczne obliczane są WYŁĄCZNIE na podstawie
# meczów rozegranych PRZED danym meczem (expanding window). Dzięki temu model
# nie ma dostępu do informacji z przyszłości.


def calculate_form(player_name, history):
    """
    Oblicza bieżącą formę gracza jako wskaźnik zwycięstw z ostatnich 10 meczów.

    Metodologia: sliding window o rozmiarze 10 meczów (bez względu na czas).
    Wartość 1.0 = 10 wygranych z rzędu, 0.0 = 10 porażek, 0.5 = brak danych (prior).

    Parametry:
        player_name: nazwa gracza
        history: DataFrame z meczami rozegranymi PRZED bieżącym meczem
    Zwraca:
        float — wskaźnik formy z zakresu [0.0, 1.0]
    """
    player_history = history[(history['winner_name'] == player_name) |
                             (history['loser_name'] == player_name)].tail(10)
    if len(player_history) == 0:
        return 0.5  # Brak historii — neutralna wartość domyślna (prior)
    wins = len(player_history[player_history['winner_name'] == player_name])
    return wins / len(player_history)


def get_h2h(p1, p2, history):
    """
    Oblicza bilans bezpośrednich pojedynków (Head-to-Head) między dwoma graczami.

    Wartość dodatnia oznacza przewagę gracza p1, ujemna — przewagę gracza p2.
    Przykład: H2H = +3 → gracz p1 wygrał 3 mecze więcej niż p2 w dotychczasowych
    bezpośrednich spotkaniach.

    Parametry:
        p1, p2: nazwy graczy
        history: DataFrame z meczami rozegranymi PRZED bieżącym meczem
    Zwraca:
        int — różnica zwycięstw p1 minus zwycięstw p2
    """
    p1_wins = len(history[(history['winner_name'] == p1) & (history['loser_name'] == p2)])
    p2_wins = len(history[(history['winner_name'] == p2) & (history['loser_name'] == p1)])
    return p1_wins - p2_wins


def calculate_surface_form(player_name, surface, history):
    """
    Oblicza formę gracza na konkretnej nawierzchni (ostatnie 10 meczów na tej nawierzchni).

    Nawierzchnia ma istotny wpływ na styl gry — niektórzy gracze dominują na korcie
    ziemnym (Clay), ale mają słabsze wyniki na trawie (Grass). Forma specyficzna
    dla nawierzchni lepiej przewiduje wynik niż forma ogólna.

    Jeśli gracz rozegrał mniej niż 3 mecze na danej nawierzchni, zwracana jest
    forma ogólna jako wartość zastępcza (fallback).

    Parametry:
        player_name: nazwa gracza
        surface: nawierzchnia meczu ('Hard', 'Clay', 'Grass')
        history: DataFrame z meczami rozegranymi PRZED bieżącym meczem
    Zwraca:
        float — wskaźnik formy na nawierzchni z zakresu [0.0, 1.0]
    """
    surface_matches = history[history['surface'] == surface]
    player_on_surface = surface_matches[
        (surface_matches['winner_name'] == player_name) |
        (surface_matches['loser_name'] == player_name)
    ].tail(10)
    if len(player_on_surface) < 3:
        return calculate_form(player_name, history)
    wins = len(player_on_surface[player_on_surface['winner_name'] == player_name])
    return wins / len(player_on_surface)


# Nazwy obliczanych statystyk serwisowych i wartości domyślne (średnie tourowe ATP).
# Używane jako prior, gdy gracz nie ma jeszcze historii meczowej.
SERVE_STAT_NAMES = ['ace_rate', 'df_rate', 'first_in_pct', 'first_won_pct',
                    'second_won_pct', 'bp_save_pct', 'bp_faced_per_game',
                    'return_pts_won']

SERVE_DEFAULTS = {
    'ace_rate': 0.08, 'df_rate': 0.03, 'first_in_pct': 0.60,
    'first_won_pct': 0.70, 'second_won_pct': 0.50,
    'bp_save_pct': 0.60, 'bp_faced_per_game': 0.40, 'return_pts_won': 0.35
}


def calculate_serve_stats(player_name, history, window=10):
    """
    Oblicza rolling average statystyk serwisowych i returnowych gracza
    z ostatnich `window` meczów.

    Dla każdego meczu w oknie sprawdza, czy gracz był zwycięzcą (kolumny w_*)
    czy przegranym (kolumny l_*), i wyciąga odpowiednie surowe statystyki.
    Następnie przelicza je na wskaźniki procentowe i uśrednia.

    Obliczane statystyki:
        ace_rate          — asy / punkty serwisowe
        df_rate           — podwójne błędy / punkty serwisowe
        first_in_pct      — % pierwszych serwisów w korcie
        first_won_pct     — % punktów wygranych na 1. serwisie
        second_won_pct    — % punktów wygranych na 2. serwisie
        bp_save_pct       — % obronionych break pointów
        bp_faced_per_game — break pointy zmierzone na gem serwisowy
        return_pts_won    — % punktów wygranych na returnie (z serwisu przeciwnika)

    Parametry:
        player_name: nazwa gracza
        history: DataFrame z meczami rozegranymi PRZED bieżącym meczem
        window: liczba ostatnich meczów do uwzględnienia (domyślnie 10)
    Zwraca:
        dict — słownik {nazwa_statystyki: wartość} z 8 wskaźnikami
    """
    player_matches = history[
        (history['winner_name'] == player_name) |
        (history['loser_name'] == player_name)
    ].tail(window)

    if len(player_matches) == 0:
        return SERVE_DEFAULTS.copy()

    ace_rates, df_rates, first_in_pcts = [], [], []
    first_won_pcts, second_won_pcts = [], []
    bp_save_pcts, bp_faced_per_games = [], []
    return_pts_won_pcts = []

    for _, match in player_matches.iterrows():
        is_winner = (match['winner_name'] == player_name)

        if is_winner:
            svpt, ace, df = match['w_svpt'], match['w_ace'], match['w_df']
            first_in, first_won = match['w_1stIn'], match['w_1stWon']
            second_won = match['w_2ndWon']
            sv_gms = match['w_SvGms']
            bp_saved, bp_faced = match['w_bpSaved'], match['w_bpFaced']
            opp_svpt = match['l_svpt']
            opp_first_won, opp_second_won = match['l_1stWon'], match['l_2ndWon']
        else:
            svpt, ace, df = match['l_svpt'], match['l_ace'], match['l_df']
            first_in, first_won = match['l_1stIn'], match['l_1stWon']
            second_won = match['l_2ndWon']
            sv_gms = match['l_SvGms']
            bp_saved, bp_faced = match['l_bpSaved'], match['l_bpFaced']
            opp_svpt = match['w_svpt']
            opp_first_won, opp_second_won = match['w_1stWon'], match['w_2ndWon']

        if svpt > 0:
            ace_rates.append(ace / svpt)
            df_rates.append(df / svpt)
            first_in_pcts.append(first_in / svpt)
        if first_in > 0:
            first_won_pcts.append(first_won / first_in)
        second_serve = svpt - first_in
        if second_serve > 0:
            second_won_pcts.append(second_won / second_serve)
        if bp_faced > 0:
            bp_save_pcts.append(bp_saved / bp_faced)
        if sv_gms > 0:
            bp_faced_per_games.append(bp_faced / sv_gms)
        if opp_svpt > 0:
            return_pts_won_pcts.append(
                (opp_svpt - opp_first_won - opp_second_won) / opp_svpt
            )

    def safe_mean(lst, default):
        return np.mean(lst) if lst else default

    return {
        'ace_rate': safe_mean(ace_rates, SERVE_DEFAULTS['ace_rate']),
        'df_rate': safe_mean(df_rates, SERVE_DEFAULTS['df_rate']),
        'first_in_pct': safe_mean(first_in_pcts, SERVE_DEFAULTS['first_in_pct']),
        'first_won_pct': safe_mean(first_won_pcts, SERVE_DEFAULTS['first_won_pct']),
        'second_won_pct': safe_mean(second_won_pcts, SERVE_DEFAULTS['second_won_pct']),
        'bp_save_pct': safe_mean(bp_save_pcts, SERVE_DEFAULTS['bp_save_pct']),
        'bp_faced_per_game': safe_mean(bp_faced_per_games, SERVE_DEFAULTS['bp_faced_per_game']),
        'return_pts_won': safe_mean(return_pts_won_pcts, SERVE_DEFAULTS['return_pts_won']),
    }


# Helpery zoptymalizowane: caller filtruje historie po graczu RAZ, dalej
# wszystkie metryki uzywaja tego samego slice'a (form / surface_form / h2h /
# serve_stats). Wczesniej kazda z tych funkcji filtrowala past_matches od nowa,
# co oznaczalo 7 niezaleznych skanow ~20k wierszy historii dla kazdego meczu.
# Te warianty oraz wektoryzacja serve_stats daja ~3-5x szybsze add_dynamic_features.

def _form_from_player_history(player_name, player_history):
    recent = player_history.tail(10)
    if len(recent) == 0:
        return 0.5
    wins = (recent['winner_name'] == player_name).sum()
    return wins / len(recent)


def _surface_form_from_player_history(player_name, surface, player_history):
    surface_matches = player_history[player_history['surface'] == surface].tail(10)
    if len(surface_matches) < 3:
        return _form_from_player_history(player_name, player_history)
    wins = (surface_matches['winner_name'] == player_name).sum()
    return wins / len(surface_matches)


def _h2h_from_p1_history(p1, p2, p1_history):
    if len(p1_history) == 0:
        return 0
    winners = p1_history['winner_name'].values
    losers = p1_history['loser_name'].values
    p1_wins = int(((winners == p1) & (losers == p2)).sum())
    p2_wins = int(((winners == p2) & (losers == p1)).sum())
    return p1_wins - p2_wins


def _serve_stats_from_player_history(player_name, player_history, window=10):
    recent = player_history.tail(window)
    if len(recent) == 0:
        return SERVE_DEFAULTS.copy()

    is_winner = (recent['winner_name'] == player_name).values

    def pick(winner_col, loser_col):
        return np.where(is_winner, recent[winner_col].values, recent[loser_col].values)

    svpt = pick('w_svpt', 'l_svpt').astype(float)
    ace = pick('w_ace', 'l_ace').astype(float)
    df_arr = pick('w_df', 'l_df').astype(float)
    first_in = pick('w_1stIn', 'l_1stIn').astype(float)
    first_won = pick('w_1stWon', 'l_1stWon').astype(float)
    second_won = pick('w_2ndWon', 'l_2ndWon').astype(float)
    sv_gms = pick('w_SvGms', 'l_SvGms').astype(float)
    bp_saved = pick('w_bpSaved', 'l_bpSaved').astype(float)
    bp_faced = pick('w_bpFaced', 'l_bpFaced').astype(float)
    opp_svpt = pick('l_svpt', 'w_svpt').astype(float)
    opp_first_won = pick('l_1stWon', 'w_1stWon').astype(float)
    opp_second_won = pick('l_2ndWon', 'w_2ndWon').astype(float)

    second_serve = svpt - first_in

    def masked_mean(numerator, denominator, default):
        valid = denominator > 0
        if not valid.any():
            return default
        return float(np.mean(numerator[valid] / denominator[valid]))

    return {
        'ace_rate': masked_mean(ace, svpt, SERVE_DEFAULTS['ace_rate']),
        'df_rate': masked_mean(df_arr, svpt, SERVE_DEFAULTS['df_rate']),
        'first_in_pct': masked_mean(first_in, svpt, SERVE_DEFAULTS['first_in_pct']),
        'first_won_pct': masked_mean(first_won, first_in, SERVE_DEFAULTS['first_won_pct']),
        'second_won_pct': masked_mean(second_won, second_serve, SERVE_DEFAULTS['second_won_pct']),
        'bp_save_pct': masked_mean(bp_saved, bp_faced, SERVE_DEFAULTS['bp_save_pct']),
        'bp_faced_per_game': masked_mean(bp_faced, sv_gms, SERVE_DEFAULTS['bp_faced_per_game']),
        'return_pts_won': masked_mean(
            opp_svpt - opp_first_won - opp_second_won,
            opp_svpt,
            SERVE_DEFAULTS['return_pts_won'],
        ),
    }


def add_dynamic_features(df_subset, historical_data):
    """
    Dołącza cechy dynamiczne (formę, formę nawierzchniową i H2H) do każdego meczu.

    Dla i-tego meczu w df_subset jako historię traktujemy:
      historical_data  +  df_subset[0..i-1]
    Dzięki temu każdy mecz „widzi" tylko przeszłość (expanding window).

    Parametry:
        df_subset: DataFrame z meczami do wzbogacenia
        historical_data: DataFrame z meczami sprzed df_subset (np. sezony 2022–2023)
    Zwraca:
        DataFrame z dodanymi kolumnami: h2h_diff, w_form, l_form, w_surface_form, l_surface_form
    """
    h2h_list = []
    w_form_list = []
    l_form_list = []
    w_sf_list = []
    l_sf_list = []
    w_serve_stats_list = []
    l_serve_stats_list = []

    full_sequence = pd.concat([historical_data, df_subset]).reset_index(drop=True)
    start_idx = len(historical_data)

    full_winner_arr = full_sequence['winner_name'].values
    full_loser_arr = full_sequence['loser_name'].values

    for i in range(len(df_subset)):
        row = df_subset.iloc[i]
        cutoff = start_idx + i
        past_matches = full_sequence.iloc[:cutoff]

        p_win = row['winner_name']
        p_los = row['loser_name']
        surface = row['surface']

        # Jedno przejscie po historii dla kazdego gracza zamiast 4 (form / surface_form / h2h / serve_stats).
        past_winners = full_winner_arr[:cutoff]
        past_losers = full_loser_arr[:cutoff]
        p_win_mask = (past_winners == p_win) | (past_losers == p_win)
        p_los_mask = (past_winners == p_los) | (past_losers == p_los)
        p_win_history = past_matches.iloc[p_win_mask.nonzero()[0]]
        p_los_history = past_matches.iloc[p_los_mask.nonzero()[0]]

        h2h_list.append(_h2h_from_p1_history(p_win, p_los, p_win_history))
        w_form_list.append(_form_from_player_history(p_win, p_win_history))
        l_form_list.append(_form_from_player_history(p_los, p_los_history))
        w_sf_list.append(_surface_form_from_player_history(p_win, surface, p_win_history))
        l_sf_list.append(_surface_form_from_player_history(p_los, surface, p_los_history))
        w_serve_stats_list.append(_serve_stats_from_player_history(p_win, p_win_history))
        l_serve_stats_list.append(_serve_stats_from_player_history(p_los, p_los_history))

    df_subset = df_subset.copy()
    df_subset['h2h_diff'] = h2h_list
    df_subset['w_form'] = w_form_list
    df_subset['l_form'] = l_form_list
    df_subset['w_surface_form'] = w_sf_list
    df_subset['l_surface_form'] = l_sf_list

    for stat_name in SERVE_STAT_NAMES:
        df_subset[f'w_{stat_name}'] = [s[stat_name] for s in w_serve_stats_list]
        df_subset[f'l_{stat_name}'] = [s[stat_name] for s in l_serve_stats_list]

    return df_subset


# Obliczanie cech dynamicznych z narastającą historią (expanding window):
#
# TRENING:    historia = sezony 2018–2023
# WALIDACJA:  historia = sezony 2018–2023 + zbiór treningowy 2024
# TEST:       historia = sezony 2018–2023 + zbiór treningowy 2024 + zbiór walidacyjny 2024
#
# W każdym przypadku przekazywane są wyłącznie kolumny bazowe (cols_base),
# aby uniknąć przypadkowego dołączenia już obliczonych cech dynamicznych.

df_train_raw = add_dynamic_features(df_train_raw, df_history_base)

history_val = pd.concat([df_history_base, df_train_raw[cols_base]]).reset_index(drop=True)
df_val_raw = add_dynamic_features(df_val_raw, history_val)

history_test = pd.concat([df_history_base, df_train_raw[cols_base], df_val_raw[cols_base]]).reset_index(drop=True)
df_test_raw = add_dynamic_features(df_test_raw, history_test)

# =============================================================================
# ETAP 6. SYMETRYZACJA DANYCH (eliminacja obciążenia pozycyjnego)
# =============================================================================
# Problem: dane źródłowe zawsze umieszczają zwycięzcę w kolumnach „winner_*",
# a przegranego w „loser_*". Gdyby model trenował bezpośrednio na takich danych,
# nauczyłby się trywialnej reguły „Gracz 1 (= winner) zawsze wygrywa" → y=1.
#
# Rozwiązanie — symetryzacja: z każdego meczu tworzone są DWA przykłady treningowe:
#   1) Gracz 1 = zwycięzca, Gracz 2 = przegrany → y = 1
#   2) Gracz 1 = przegrany, Gracz 2 = zwycięzca → y = 0
# Dzięki temu rozkład etykiet jest idealnie zbalansowany (50/50),
# a model uczy się rozpoznawać, KTÓRY z dwóch graczy wygra — nie „czy Gracz 1 wygra".


def symmetrize_data(df_subset, shuffle=True):
    """
    Tworzy symetryczny (zbalansowany) zbiór danych z zamianą ról graczy.

    Dla każdego meczu generowane są dwa wiersze z odwróconymi perspektywami.
    Cechy zależne od gracza (ranking, forma, H2H) są odpowiednio zamieniane.

    Parametry:
        df_subset: DataFrame z meczami (po obliczeniu cech dynamicznych)
        shuffle: True = losowe wymieszanie (trening/ewaluacja),
                 False = zachowanie kolejności chronologicznej (walidacja krzyżowa)
    Zwraca:
        DataFrame o podwojonej liczbie wierszy z kolumnami p1_*/p2_* oraz etykietą y
    """
    rows_p1_wins = []
    rows_p2_wins = []

    for idx, row in df_subset.iterrows():
        # Perspektywa 1: Gracz 1 = zwycięzca meczu → etykieta y = 1
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
            'p1_form': row['w_form'],
            'p2_form': row['l_form'],
            'p1_surface_form': row['w_surface_form'],
            'p2_surface_form': row['l_surface_form'],
            'rank_diff': row['winner_rank_log'] - row['loser_rank_log'],
            'rank_pts_diff': row['winner_rank_pts_log'] - row['loser_rank_pts_log'],
            'age_diff': row['winner_age'] - row['loser_age'],
            'ht_diff': row['winner_ht'] - row['loser_ht'],
            'form_diff': row['w_form'] - row['l_form'],
            'p1_ace_rate': row['w_ace_rate'],
            'p2_ace_rate': row['l_ace_rate'],
            'p1_df_rate': row['w_df_rate'],
            'p2_df_rate': row['l_df_rate'],
            'p1_first_in_pct': row['w_first_in_pct'],
            'p2_first_in_pct': row['l_first_in_pct'],
            'p1_first_won_pct': row['w_first_won_pct'],
            'p2_first_won_pct': row['l_first_won_pct'],
            'p1_second_won_pct': row['w_second_won_pct'],
            'p2_second_won_pct': row['l_second_won_pct'],
            'p1_bp_save_pct': row['w_bp_save_pct'],
            'p2_bp_save_pct': row['l_bp_save_pct'],
            'p1_bp_faced_per_game': row['w_bp_faced_per_game'],
            'p2_bp_faced_per_game': row['l_bp_faced_per_game'],
            'p1_return_pts_won': row['w_return_pts_won'],
            'p2_return_pts_won': row['l_return_pts_won'],
            'y': 1,
            'actual_winner': row['winner_name'],
            'actual_loser': row['loser_name'],
            'p1_name': row['winner_name'],
            'p2_name': row['loser_name']
        }

        # Perspektywa 2: Gracz 1 = przegrany meczu → etykieta y = 0
        # Cechy różnicowe i H2H są negowane (perspektywa odwrócona).
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
            'p1_form': row['l_form'],
            'p2_form': row['w_form'],
            'p1_surface_form': row['l_surface_form'],
            'p2_surface_form': row['w_surface_form'],
            'rank_diff': row['loser_rank_log'] - row['winner_rank_log'],
            'rank_pts_diff': row['loser_rank_pts_log'] - row['winner_rank_pts_log'],
            'age_diff': row['loser_age'] - row['winner_age'],
            'ht_diff': row['loser_ht'] - row['winner_ht'],
            'form_diff': row['l_form'] - row['w_form'],
            'p1_ace_rate': row['l_ace_rate'],
            'p2_ace_rate': row['w_ace_rate'],
            'p1_df_rate': row['l_df_rate'],
            'p2_df_rate': row['w_df_rate'],
            'p1_first_in_pct': row['l_first_in_pct'],
            'p2_first_in_pct': row['w_first_in_pct'],
            'p1_first_won_pct': row['l_first_won_pct'],
            'p2_first_won_pct': row['w_first_won_pct'],
            'p1_second_won_pct': row['l_second_won_pct'],
            'p2_second_won_pct': row['w_second_won_pct'],
            'p1_bp_save_pct': row['l_bp_save_pct'],
            'p2_bp_save_pct': row['w_bp_save_pct'],
            'p1_bp_faced_per_game': row['l_bp_faced_per_game'],
            'p2_bp_faced_per_game': row['w_bp_faced_per_game'],
            'p1_return_pts_won': row['l_return_pts_won'],
            'p2_return_pts_won': row['w_return_pts_won'],
            'y': 0,
            'actual_winner': row['winner_name'],
            'actual_loser': row['loser_name'],
            'p1_name': row['loser_name'],
            'p2_name': row['winner_name']
        }

        rows_p1_wins.append(row1)
        rows_p2_wins.append(row2)

    # Przeplatanie: para (perspektywa 1, perspektywa 2) dla każdego meczu
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
# ETAP 7. DEFINICJA WEKTORA CECH (feature vector)
# =============================================================================
# Wektor cech wejściowych modelu (40 cech). Każdy wiersz opisuje parę graczy w jednym meczu:
#   - surface, tourney_level, best_of, round_num — kontekst meczu
#   - p1/p2_rank_log, p1/p2_rank_pts_log         — ranking ATP i punkty (log)
#   - p1/p2_age, p1/p2_ht                         — wiek i wzrost obu graczy
#   - p1/p2_is_lefty                               — ręczność (1 = leworęczny)
#   - p1_h2h                                       — bilans H2H z perspektywy Gracza 1
#   - p1/p2_form, p1/p2_surface_form               — forma ogólna i nawierzchniowa
#   - p1/p2_*_rate/pct                             — rolling serwis/return (8 per gracz)
#   - rank_diff, rank_pts_diff, age_diff, ht_diff, form_diff — cechy różnicowe

features = ['surface', 'tourney_level', 'best_of', 'round_num',
            'p1_rank_log', 'p1_rank_pts_log', 'p1_age', 'p1_ht', 'p1_is_lefty',
            'p2_rank_log', 'p2_rank_pts_log', 'p2_age', 'p2_ht', 'p2_is_lefty',
            'p1_h2h', 'p1_form', 'p2_form',
            'p1_surface_form', 'p2_surface_form',
            'p1_ace_rate', 'p2_ace_rate', 'p1_df_rate', 'p2_df_rate',
            'p1_first_in_pct', 'p2_first_in_pct',
            'p1_first_won_pct', 'p2_first_won_pct',
            'p1_second_won_pct', 'p2_second_won_pct',
            'p1_bp_save_pct', 'p2_bp_save_pct',
            'p1_bp_faced_per_game', 'p2_bp_faced_per_game',
            'p1_return_pts_won', 'p2_return_pts_won',
            'rank_diff', 'rank_pts_diff', 'age_diff', 'ht_diff', 'form_diff']

X_val = val_data[features]
y_val = val_data['y']

X_test = test_data[features]
y_test = test_data['y']


# =============================================================================
# ETAP 8. OPTYMALIZACJA HIPERPARAMETRÓW (RandomizedSearchCV + TimeSeriesSplit)
# =============================================================================
# Algorytm Random Forest ma wiele hiperparametrów (głębokość drzew, liczba drzew,
# minimalne próbki do podziału itp.). RandomizedSearchCV losowo próbkuje kombinacje
# z ustalonej przestrzeni i ocenia je za pomocą walidacji krzyżowej.
#
# TimeSeriesSplit dzieli dane treningowe na 5 foldów z zachowaniem chronologii:
#   Fold 1: train=[0..N/5],      val=[N/5..2N/5]
#   Fold 2: train=[0..2N/5],     val=[2N/5..3N/5]
#   ...itd.
# Dzięki temu model w żadnym foldzie nie trenuje na danych „z przyszłości".
#
# Dane treningowe dla CV NIE są losowo mieszane (shuffle=False), aby zachować
# porządek chronologiczny wymagany przez TimeSeriesSplit.

from sklearn.model_selection import TimeSeriesSplit

train_data_ordered = symmetrize_data(df_train_raw, shuffle=False)
X_train_cv = train_data_ordered[features]
y_train_cv = train_data_ordered['y']

print(f"Próbki treningowe dla walidacji krzyżowej: {len(train_data_ordered)}")

param_dist = {
    'n_estimators': [100, 200, 300, 500],       # Liczba drzew w lesie
    'max_depth': [10, 15, 20, 30, None],         # Maksymalna głębokość drzewa
    'min_samples_split': [2, 5, 10, 20],         # Min. próbek do podziału węzła
    'min_samples_leaf': [1, 2, 4, 8],            # Min. próbek w liściu
    'max_features': ['sqrt', 'log2'],            # Liczba cech losowanych przy podziale
    'bootstrap': [True],                         # Próbkowanie bootstrapowe (z powtórzeniami)
    'max_samples': [0.7, 0.8, 0.9, 1.0]         # Odsetek próbek w każdym drzewie (regularyzacja)
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
best_rf.n_jobs = -1  # Przywrócenie pełnej równoległości dla finalnego treningu

# Finalny model trenowany na pełnym zbiorze treningowym (z losowym wymieszaniem).
# Dane są mieszane, ponieważ nie wykonujemy już walidacji krzyżowej —
# Random Forest korzysta z wewnętrznego baggingu, który sam losuje podzbiory.
train_data_final = symmetrize_data(df_train_raw, shuffle=True)
X_train_final = train_data_final[features]
y_train_final = train_data_final['y']

print(f"Trening finalnego modelu na {len(train_data_final)} próbkach...")
best_rf.fit(X_train_final, y_train_final)

# =============================================================================
# ETAP 9. EWALUACJA MODELU
# =============================================================================
# Ewaluacja na dwóch zbiorach:
#   - Walidacyjnym (20% środkowych danych 2024) — do kontroli podczas eksperymentów
#   - Testowym (20% najnowszych danych 2024) — ostateczna ocena jakości
#
# Metryki:
#   - Accuracy — odsetek poprawnych klasyfikacji (na symetryzowanych danych)
#   - Classification Report — precision, recall, F1-score per klasa
#   - Confusion Matrix — macierz pomyłek

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
# Symetryzacja powoduje, że każdy mecz ma dwa wiersze w zbiorze testowym.
# Aby uzyskać jedną predykcję na mecz, wybieramy unikalne mecze (po match_id)
# i sprawdzamy, czy przewidziany zwycięzca zgadza się z rzeczywistym.

print("\n" + "="*50)
print("=== PRZEWIDYWANIE ZWYCIĘZCÓW MECZÓW ===")
print("="*50)

test_data['p1_win_probability'] = test_pred_proba[:, 1]

# Ewaluacja na poziomie meczów: z każdej pary symetrycznej bierzemy perspektywę
# zwycięzcy (y=1, gdzie p1 = rzeczywisty zwycięzca). Sprawdzamy, czy model
# przypisał zwycięzcy prawdopodobieństwo > 0.5.
# Ta metoda jest deterministyczna (nie zależy od kolejności po shuffle).
winner_perspective = test_data[test_data['y'] == 1].copy()
winner_perspective['predicted_winner'] = winner_perspective.apply(
    lambda row: row['p1_name'] if row['p1_win_probability'] > 0.5 else row['p2_name'],
    axis=1
)
winner_perspective['correct_prediction'] = winner_perspective['p1_win_probability'] > 0.5
match_accuracy = winner_perspective['correct_prediction'].mean()

print(f"\nACCURACY PRZEWIDYWANIA ZWYCIEZCOW: {match_accuracy:.4f} ({match_accuracy*100:.2f}%)")
print(f"Poprawnie przewidziane: {int(winner_perspective['correct_prediction'].sum())} / {len(winner_perspective)} meczow")

print("\nPrzykladowe predykcje (pierwsze 10 meczow):")
print("-" * 80)
sample_predictions = winner_perspective.head(10)[['p1_name', 'p2_name', 'actual_winner',
                                                   'predicted_winner', 'p1_win_probability',
                                                   'correct_prediction']]
for idx, row in sample_predictions.iterrows():
    status = " OK " if row['correct_prediction'] else "MISS"
    print(f"[{status}] {row['p1_name'][:20]:20s} vs {row['p2_name'][:20]:20s}")
    print(f"       Rzeczywisty: {row['actual_winner'][:25]:25s} | Przewidziany: {row['predicted_winner'][:25]:25s}")
    print(f"       P(zwyciezca wygra): {row['p1_win_probability']:.3f}")
    print()

# =============================================================================
# ETAP 11. WAŻNOŚĆ CECH (Feature Importance)
# =============================================================================
# Random Forest oblicza ważność cech jako średni spadek zanieczyszczenia Giniego
# (Mean Decrease Impurity) uśredniony po wszystkich drzewach. Cechy z wyższą
# wartością mają większy wpływ na decyzje klasyfikacyjne modelu.

print("\n=== WAŻNOŚĆ CECH ===")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)


# =============================================================================
# ETAP 12. ANALIZA BŁĘDÓW (najgorsze predykcje)
# =============================================================================
# Analiza przypadków, w których model był najbardziej pewny siebie, ale się mylił.
# Błąd mierzony jako |prawdopodobieństwo przypisane rzeczywistemu zwycięzcy - 1|.
# Wysokie wartości błędu wskazują na sytuacje, w których cechy modelu nie uchwyciły
# prawdziwej dynamiki meczu (np. kontuzja, zmiana formy, debiut na nawierzchni).

print("\n" + "="*50)
print("=== ANALIZA NAJWIĘKSZYCH BŁĘDÓW ===")
print("="*50)

# W winner_perspective p1 = rzeczywisty zwycięzca, więc błąd = 1 - P(p1 wygra).
# Wysoki błąd = model był pewny, że zwycięzca przegra.
winner_perspective['error'] = 1 - winner_perspective['p1_win_probability']

worst_predictions = winner_perspective.nlargest(5, 'error')[['p1_name', 'p2_name', 'actual_winner',
                                                              'predicted_winner', 'p1_win_probability', 'error']]
print("\nTop 5 najgorszych predykcji:")
for idx, row in worst_predictions.iterrows():
    print(f"Mecz: {row['p1_name']} vs {row['p2_name']}")
    print(f"  Rzeczywisty zwyciezca: {row['actual_winner']}")
    print(f"  Przewidziany: {row['predicted_winner']} (P(zwyciezca): {row['p1_win_probability']:.3f})")
    print(f"  Blad: {row['error']:.3f}\n")


# =============================================================================
# ETAP 13. PODSUMOWANIE WYNIKÓW
# =============================================================================

print("\n" + "="*50)
print("=== PODSUMOWANIE ===")
print("="*50)
print(f"Statystyki modelu:")
print(f"CV Accuracy:         {search.best_score_:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Match Prediction:    {match_accuracy:.4f}")
print("\nBaseline (losowe): 0.5000 (50%)")
print(f"Mój model:        {match_accuracy:.4f} ({match_accuracy*100:.1f}%)")
print(f"Przewaga nad losowym zgadywaniem: +{(match_accuracy-0.5)*100:.1f} p.p.")