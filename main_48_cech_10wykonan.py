"""
Test wielokrotny modelu main.py (wersja rozszerzona — 40 cech)
==============================================================
Uruchamia model 10 razy z różnymi ziarnami losowości (RANDOM_STATE),
zbiera Match Prediction Accuracy z każdego przebiegu i wylicza średnią.

Cechy dynamiczne (forma, H2H, serwis) obliczane są RAZ — nie zależą od ziarna.
Ziarno wpływa wyłącznie na: RandomizedSearchCV, symetryzację (shuffle), Random Forest.
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import time

N_RUNS = 10
SEEDS = list(range(1, N_RUNS + 1))  # Ziarna: 1, 2, ..., 10


# =============================================================================
# ETAP 1–5: PRZYGOTOWANIE DANYCH (jednorazowe — niezależne od ziarna)
# =============================================================================

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
df_base['winner_rank_log'] = np.log(df_base['winner_rank'])
df_base['loser_rank_log'] = np.log(df_base['loser_rank'])
df_base['winner_rank_pts_log'] = np.log(df_base['winner_rank_points'])
df_base['loser_rank_pts_log'] = np.log(df_base['loser_rank_points'])
df_base['winner_is_lefty'] = (df_base['winner_hand'] == 'L').astype(int)
df_base['loser_is_lefty'] = (df_base['loser_hand'] == 'L').astype(int)

ROUND_ORDER = {'R128': 1, 'R64': 2, 'R32': 3, 'RR': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'BR': 6, 'F': 7}
df_base['round_encoded'] = df_base['round'].map(ROUND_ORDER).fillna(3)

print(f"Dane główne (2024): {len(df_base)} meczów")

# Historia
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
    except FileNotFoundError:
        pass
df_history_base = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame(columns=cols_base)
print(f"Historia: {len(df_history_base)} meczów")

# Label Encoding
le_surface = LabelEncoder()
le_level = LabelEncoder()
all_surfaces = pd.concat([df_base['surface'], df_history_base['surface']]).unique()
all_levels = pd.concat([df_base['tourney_level'], df_history_base['tourney_level']]).unique()
le_surface.fit(all_surfaces)
le_level.fit(all_levels)
df_base['surface_encoded'] = le_surface.transform(df_base['surface'])
df_base['tourney_level_encoded'] = le_level.transform(df_base['tourney_level'])

# Podział chronologiczny
train_end = int(len(df_base) * 0.60)
val_end = int(len(df_base) * 0.80)
df_train_raw = df_base.iloc[:train_end].reset_index(drop=True)
df_val_raw = df_base.iloc[train_end:val_end].reset_index(drop=True)
df_test_raw = df_base.iloc[val_end:].reset_index(drop=True)
df_train_raw['match_id'] = range(len(df_train_raw))
df_val_raw['match_id'] = range(len(df_val_raw))
df_test_raw['match_id'] = range(len(df_test_raw))


# --- Funkcje cech dynamicznych ---

def calculate_form(player_name, history):
    player_history = history[(history['winner_name'] == player_name) |
                             (history['loser_name'] == player_name)].tail(10)
    if len(player_history) == 0:
        return 0.5
    wins = len(player_history[player_history['winner_name'] == player_name])
    return wins / len(player_history)


def get_h2h(p1, p2, history):
    p1_wins = len(history[(history['winner_name'] == p1) & (history['loser_name'] == p2)])
    p2_wins = len(history[(history['winner_name'] == p2) & (history['loser_name'] == p1)])
    return p1_wins - p2_wins


def calculate_surface_form(player_name, surface, history):
    surface_matches = history[history['surface'] == surface]
    player_on_surface = surface_matches[
        (surface_matches['winner_name'] == player_name) |
        (surface_matches['loser_name'] == player_name)
    ].tail(10)
    if len(player_on_surface) < 3:
        return calculate_form(player_name, history)
    wins = len(player_on_surface[player_on_surface['winner_name'] == player_name])
    return wins / len(player_on_surface)


SERVE_STAT_NAMES = ['ace_rate', 'df_rate', 'first_in_pct', 'first_won_pct',
                    'second_won_pct', 'bp_save_pct', 'bp_faced_per_game', 'return_pts_won']

SERVE_DEFAULTS = {
    'ace_rate': 0.08, 'df_rate': 0.03, 'first_in_pct': 0.60,
    'first_won_pct': 0.70, 'second_won_pct': 0.50,
    'bp_save_pct': 0.60, 'bp_faced_per_game': 0.40, 'return_pts_won': 0.35
}


def calculate_serve_stats(player_name, history, window=10):
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
            svpt, ace, df_ = match['w_svpt'], match['w_ace'], match['w_df']
            first_in, first_won = match['w_1stIn'], match['w_1stWon']
            second_won = match['w_2ndWon']
            sv_gms = match['w_SvGms']
            bp_saved, bp_faced = match['w_bpSaved'], match['w_bpFaced']
            opp_svpt = match['l_svpt']
            opp_first_won, opp_second_won = match['l_1stWon'], match['l_2ndWon']
        else:
            svpt, ace, df_ = match['l_svpt'], match['l_ace'], match['l_df']
            first_in, first_won = match['l_1stIn'], match['l_1stWon']
            second_won = match['l_2ndWon']
            sv_gms = match['l_SvGms']
            bp_saved, bp_faced = match['l_bpSaved'], match['l_bpFaced']
            opp_svpt = match['w_svpt']
            opp_first_won, opp_second_won = match['w_1stWon'], match['w_2ndWon']

        if svpt > 0:
            ace_rates.append(ace / svpt)
            df_rates.append(df_ / svpt)
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
            return_pts_won_pcts.append((opp_svpt - opp_first_won - opp_second_won) / opp_svpt)

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


def add_dynamic_features(df_subset, historical_data):
    h2h_list, w_form_list, l_form_list = [], [], []
    w_sf_list, l_sf_list = [], []
    w_serve_list, l_serve_list = [], []

    full_sequence = pd.concat([historical_data, df_subset]).reset_index(drop=True)
    start_idx = len(historical_data)

    for i in range(len(df_subset)):
        row = df_subset.iloc[i]
        past = full_sequence.iloc[:start_idx + i]
        p_w, p_l, surf = row['winner_name'], row['loser_name'], row['surface']

        h2h_list.append(get_h2h(p_w, p_l, past))
        w_form_list.append(calculate_form(p_w, past))
        l_form_list.append(calculate_form(p_l, past))
        w_sf_list.append(calculate_surface_form(p_w, surf, past))
        l_sf_list.append(calculate_surface_form(p_l, surf, past))
        w_serve_list.append(calculate_serve_stats(p_w, past))
        l_serve_list.append(calculate_serve_stats(p_l, past))

    df_subset = df_subset.copy()
    df_subset['h2h_diff'] = h2h_list
    df_subset['w_form'] = w_form_list
    df_subset['l_form'] = l_form_list
    df_subset['w_surface_form'] = w_sf_list
    df_subset['l_surface_form'] = l_sf_list
    for s in SERVE_STAT_NAMES:
        df_subset[f'w_{s}'] = [x[s] for x in w_serve_list]
        df_subset[f'l_{s}'] = [x[s] for x in l_serve_list]
    return df_subset


print("Obliczanie cech dynamicznych (jednorazowe)...")
t0 = time.time()
df_train_raw = add_dynamic_features(df_train_raw, df_history_base)
history_val = pd.concat([df_history_base, df_train_raw[cols_base]]).reset_index(drop=True)
df_val_raw = add_dynamic_features(df_val_raw, history_val)
history_test = pd.concat([df_history_base, df_train_raw[cols_base], df_val_raw[cols_base]]).reset_index(drop=True)
df_test_raw = add_dynamic_features(df_test_raw, history_test)
print(f"Cechy dynamiczne obliczone w {time.time()-t0:.1f}s")


# --- Symetryzacja i model (zależne od ziarna) ---

def symmetrize_data(df_subset, seed, shuffle=True):
    rows_p1_wins, rows_p2_wins = [], []
    for idx, row in df_subset.iterrows():
        row1 = {
            'match_id': row['match_id'],
            'surface': row['surface_encoded'], 'tourney_level': row['tourney_level_encoded'],
            'best_of': row['best_of'], 'round_num': row['round_encoded'],
            'p1_rank_log': row['winner_rank_log'], 'p1_rank_pts_log': row['winner_rank_pts_log'],
            'p1_age': row['winner_age'], 'p1_ht': row['winner_ht'], 'p1_is_lefty': row['winner_is_lefty'],
            'p2_rank_log': row['loser_rank_log'], 'p2_rank_pts_log': row['loser_rank_pts_log'],
            'p2_age': row['loser_age'], 'p2_ht': row['loser_ht'], 'p2_is_lefty': row['loser_is_lefty'],
            'p1_h2h': row['h2h_diff'],
            'p1_form': row['w_form'], 'p2_form': row['l_form'],
            'p1_surface_form': row['w_surface_form'], 'p2_surface_form': row['l_surface_form'],
            'rank_diff': row['winner_rank_log'] - row['loser_rank_log'],
            'rank_pts_diff': row['winner_rank_pts_log'] - row['loser_rank_pts_log'],
            'age_diff': row['winner_age'] - row['loser_age'],
            'ht_diff': row['winner_ht'] - row['loser_ht'],
            'form_diff': row['w_form'] - row['l_form'],
            'y': 1, 'actual_winner': row['winner_name'], 'actual_loser': row['loser_name'],
            'p1_name': row['winner_name'], 'p2_name': row['loser_name']
        }
        for s in SERVE_STAT_NAMES:
            row1[f'p1_{s}'] = row[f'w_{s}']
            row1[f'p2_{s}'] = row[f'l_{s}']

        row2 = {
            'match_id': row['match_id'],
            'surface': row['surface_encoded'], 'tourney_level': row['tourney_level_encoded'],
            'best_of': row['best_of'], 'round_num': row['round_encoded'],
            'p1_rank_log': row['loser_rank_log'], 'p1_rank_pts_log': row['loser_rank_pts_log'],
            'p1_age': row['loser_age'], 'p1_ht': row['loser_ht'], 'p1_is_lefty': row['loser_is_lefty'],
            'p2_rank_log': row['winner_rank_log'], 'p2_rank_pts_log': row['winner_rank_pts_log'],
            'p2_age': row['winner_age'], 'p2_ht': row['winner_ht'], 'p2_is_lefty': row['winner_is_lefty'],
            'p1_h2h': -row['h2h_diff'],
            'p1_form': row['l_form'], 'p2_form': row['w_form'],
            'p1_surface_form': row['l_surface_form'], 'p2_surface_form': row['w_surface_form'],
            'rank_diff': row['loser_rank_log'] - row['winner_rank_log'],
            'rank_pts_diff': row['loser_rank_pts_log'] - row['winner_rank_pts_log'],
            'age_diff': row['loser_age'] - row['winner_age'],
            'ht_diff': row['loser_ht'] - row['winner_ht'],
            'form_diff': row['l_form'] - row['w_form'],
            'y': 0, 'actual_winner': row['winner_name'], 'actual_loser': row['loser_name'],
            'p1_name': row['loser_name'], 'p2_name': row['winner_name']
        }
        for s in SERVE_STAT_NAMES:
            row2[f'p1_{s}'] = row[f'l_{s}']
            row2[f'p2_{s}'] = row[f'w_{s}']

        rows_p1_wins.append(row1)
        rows_p2_wins.append(row2)

    all_rows = []
    for r1, r2 in zip(rows_p1_wins, rows_p2_wins):
        all_rows.extend([r1, r2])
    result = pd.DataFrame(all_rows)
    if shuffle:
        result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)
    return result


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

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'max_samples': [0.7, 0.8, 0.9, 1.0]
}


# =============================================================================
# PĘTLA GŁÓWNA — 10 uruchomień z różnymi ziarnami
# =============================================================================

results = []

print(f"\n{'='*60}")
print(f"  MAIN.PY (40 cech) — TEST {N_RUNS} URUCHOMIEŃ")
print(f"{'='*60}\n")

for run_idx, seed in enumerate(SEEDS, 1):
    t_start = time.time()
    print(f"--- Przebieg {run_idx}/{N_RUNS} (seed={seed}) ---")

    # Symetryzacja z bieżącym ziarnem
    val_data = symmetrize_data(df_val_raw, seed, shuffle=True)
    test_data = symmetrize_data(df_test_raw, seed, shuffle=True)
    train_data_ordered = symmetrize_data(df_train_raw, seed, shuffle=False)
    train_data_final = symmetrize_data(df_train_raw, seed, shuffle=True)

    X_train_cv = train_data_ordered[features]
    y_train_cv = train_data_ordered['y']
    X_val = val_data[features]
    y_val = val_data['y']
    X_test = test_data[features]
    y_test = test_data['y']

    # RandomizedSearchCV z bieżącym ziarnem
    rf = RandomForestClassifier(n_jobs=1, random_state=seed)
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(rf, param_dist, n_iter=50, cv=tscv,
                                scoring='accuracy', n_jobs=-1, verbose=0,
                                random_state=seed)
    search.fit(X_train_cv, y_train_cv)

    best_rf = search.best_estimator_
    best_rf.n_jobs = -1

    # Trening finalny
    X_train_final = train_data_final[features]
    y_train_final = train_data_final['y']
    best_rf.fit(X_train_final, y_train_final)

    # Ewaluacja
    val_acc = accuracy_score(y_val, best_rf.predict(X_val))
    test_pred_proba = best_rf.predict_proba(X_test)
    test_acc = accuracy_score(y_test, best_rf.predict(X_test))

    # Match accuracy
    test_data_copy = test_data.copy()
    test_data_copy['p1_win_prob'] = test_pred_proba[:, 1]
    wp = test_data_copy[test_data_copy['y'] == 1].copy()
    wp['correct'] = wp['p1_win_prob'] > 0.5
    match_acc = wp['correct'].mean()

    elapsed = time.time() - t_start
    results.append({
        'seed': seed,
        'cv_acc': search.best_score_,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'match_acc': match_acc,
        'best_params': search.best_params_,
        'time_s': elapsed
    })
    print(f"  CV={search.best_score_:.4f}  Val={val_acc:.4f}  Test={test_acc:.4f}  "
          f"Match={match_acc:.4f} ({match_acc*100:.2f}%)  [{elapsed:.1f}s]")


# =============================================================================
# PODSUMOWANIE
# =============================================================================

print(f"\n{'='*60}")
print(f"  PODSUMOWANIE — MAIN.PY (40 cech, {N_RUNS} uruchomień)")
print(f"{'='*60}\n")

cv_scores = [r['cv_acc'] for r in results]
val_scores = [r['val_acc'] for r in results]
test_scores = [r['test_acc'] for r in results]
match_scores = [r['match_acc'] for r in results]

print(f"{'Przebieg':>10} {'Seed':>6} {'CV Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Match Acc':>12}")
print("-" * 62)
for r in results:
    print(f"{'':>10} {r['seed']:>6} {r['cv_acc']:>10.4f} {r['val_acc']:>10.4f} "
          f"{r['test_acc']:>10.4f} {r['match_acc']:>10.4f} ({r['match_acc']*100:.2f}%)")

print("-" * 62)
print(f"{'ŚREDNIA':>10} {'':>6} {np.mean(cv_scores):>10.4f} {np.mean(val_scores):>10.4f} "
      f"{np.mean(test_scores):>10.4f} {np.mean(match_scores):>10.4f} ({np.mean(match_scores)*100:.2f}%)")
print(f"{'STD':>10} {'':>6} {np.std(cv_scores):>10.4f} {np.std(val_scores):>10.4f} "
      f"{np.std(test_scores):>10.4f} {np.std(match_scores):>10.4f}")
print(f"{'MIN':>10} {'':>6} {np.min(cv_scores):>10.4f} {np.min(val_scores):>10.4f} "
      f"{np.min(test_scores):>10.4f} {np.min(match_scores):>10.4f} ({np.min(match_scores)*100:.2f}%)")
print(f"{'MAX':>10} {'':>6} {np.max(cv_scores):>10.4f} {np.max(val_scores):>10.4f} "
      f"{np.max(test_scores):>10.4f} {np.max(match_scores):>10.4f} ({np.max(match_scores)*100:.2f}%)")

total_time = sum(r['time_s'] for r in results)
print(f"\nŁączny czas: {total_time:.1f}s (średnio {total_time/N_RUNS:.1f}s / przebieg)")
