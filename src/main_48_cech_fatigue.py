"""
Eksperyment Sprint 3b: Cechy zmeczenia (fatigue)
================================================

Dwie nowe cechy per gracz, liczone ze scisle wczesniejszych meczow (bez leakage):

1) rest_days -- dni od ostatniego meczu gracza. Krotki odpoczynek = zmeczenie,
   bardzo dlugi = brak rytmu / powrot po kontuzji. Cap na MAX_REST dni.

2) tourney_minutes -- suma minut zagranych przez gracza w BIEZACYM turnieju
   (wczesniejsze rundy). Gracz, ktory przeszedl przez 3 dlugie pieciosetowki,
   wchodzi w cwiercfinal bardziej zmeczony niz ktos po szybkich zwyciestwach.

Cechy symetryzowane do p1_/p2_ + roznice. Porownanie baseline RF vs
baseline + fatigue (ablation, te same tuned HP, metryka symetryczna z Sprint 1).
"""

from __future__ import annotations

import bisect
import contextlib
import io
import os
import runpy
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


BASE_SCRIPT = Path(__file__).with_name("main_48_cech.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
TOUR = os.environ.get("TENNIS_TOUR", "atp")
TARGET_YEAR = int(os.environ.get("TENNIS_TARGET_YEAR", "2025"))
HISTORY_START_YEAR = int(os.environ.get("TENNIS_HISTORY_START", "2001"))


def data_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


HISTORY_FILES = [data_file(y) for y in range(HISTORY_START_YEAR, TARGET_YEAR)]

MAX_REST = 60.0       # dni; powyzej tego "wypoczety" sie nasyca
DEFAULT_REST = 60.0   # gdy brak wczesniejszego meczu (np. pierwszy mecz w danych)

NEW_FEATURES = [
    "p1_rest_days", "p2_rest_days", "rest_days_diff",
    "p1_tourney_minutes", "p2_tourney_minutes", "tourney_minutes_diff",
]

FATIGUE_COLS = ["winner_name", "loser_name", "tourney_date", "tourney_id", "minutes"]


def execute_base_pipeline_quietly() -> dict:
    original_cwd = os.getcwd()
    captured = io.StringIO()
    os.chdir(BASE_SCRIPT.parent)
    try:
        with contextlib.redirect_stdout(captured):
            return runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)


def load_fatigue_frame(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    return df[FATIGUE_COLS].copy()


def build_player_index(seq: pd.DataFrame) -> dict[str, list[int]]:
    winners = seq["winner_name"].to_numpy()
    losers = seq["loser_name"].to_numpy()
    idx: dict[str, list[int]] = defaultdict(list)
    for i in range(len(seq)):
        idx[winners[i]].append(i)
        if losers[i] != winners[i]:
            idx[losers[i]].append(i)
    return idx


def compute_fatigue_for_2024(
    df_2024_base: pd.DataFrame,
    history_files=None,
) -> pd.DataFrame:
    """Zwraca ramke z kolumnami w_rest_days/l_rest_days/w_tourney_minutes/
    l_tourney_minutes wyrownana 1:1 do df_2024_base (po pozycji).

    history_files: None -> domyslne 2018-2023. Dla walk-forward przekazujemy
    historie wlasciwa dla roku docelowego."""
    if history_files is None:
        history_files = HISTORY_FILES
    history = pd.concat([load_fatigue_frame(p) for p in history_files], ignore_index=True)
    full_seq = pd.concat([history, df_2024_base[FATIGUE_COLS]], ignore_index=True)
    start_idx = len(history)

    dates = full_seq["tourney_date"].to_numpy()
    tourneys = full_seq["tourney_id"].to_numpy()
    minutes = pd.to_numeric(full_seq["minutes"], errors="coerce").to_numpy()
    winners = full_seq["winner_name"].to_numpy()
    losers = full_seq["loser_name"].to_numpy()

    player_index = build_player_index(full_seq)

    def player_stats(name, cutoff, current_date, current_tourney):
        all_idx = player_index.get(name, [])
        end = bisect.bisect_left(all_idx, cutoff)
        earlier = all_idx[:end]
        if not earlier:
            return DEFAULT_REST, 0.0
        # rest_days: od najswiezszego wczesniejszego meczu (earlier jest sorted).
        last_idx = earlier[-1]
        rest = (current_date - dates[last_idx]) / np.timedelta64(1, "D")
        rest = float(min(max(rest, 0.0), MAX_REST))
        # tourney_minutes: suma minut w tym samym turnieju (wczesniejsze rundy).
        tmin = 0.0
        for j in earlier:
            if tourneys[j] == current_tourney:
                m = minutes[j]
                if not np.isnan(m):
                    tmin += float(m)
        return rest, tmin

    w_rest, l_rest, w_min, l_min = [], [], [], []
    for i in range(len(df_2024_base)):
        cutoff = start_idx + i
        cur_date = dates[cutoff]
        cur_tourney = tourneys[cutoff]
        p_win = winners[cutoff]
        p_los = losers[cutoff]
        wr, wm = player_stats(p_win, cutoff, cur_date, cur_tourney)
        lr, lm = player_stats(p_los, cutoff, cur_date, cur_tourney)
        w_rest.append(wr); l_rest.append(lr); w_min.append(wm); l_min.append(lm)

    return pd.DataFrame({
        "w_rest_days": w_rest, "l_rest_days": l_rest,
        "w_tourney_minutes": w_min, "l_tourney_minutes": l_min,
    })


def main() -> None:
    ns = execute_base_pipeline_quietly()
    symmetrize_data = ns["symmetrize_data"]
    compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
    evaluate_calibration_quality = ns["evaluate_calibration_quality"]
    baseline_search = ns["search"]
    RANDOM_STATE = ns["RANDOM_STATE"]
    base_features = list(ns["features"])
    cols_base = list(ns["cols_base"])

    df_train_raw = ns["df_train_raw"].copy()
    df_val_raw = ns["df_val_raw"].copy()
    df_test_raw = ns["df_test_raw"].copy()
    baseline_val_acc = float(ns["val_acc"])
    baseline_test_acc = float(ns["test_acc"])
    baseline_match_acc = float(ns["match_accuracy"])

    # Odtworz pelny rok docelowy (z tourney_id + minutes), te same wiersze i kolejnosc co baseline.
    print(f"Licze cechy zmeczenia (rest_days, tourney_minutes) z historii + {TARGET_YEAR}...")
    full_target = pd.read_csv(data_file(TARGET_YEAR))
    full_target["tourney_date"] = pd.to_datetime(full_target["tourney_date"], format="%Y%m%d")
    full_target = full_target.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    full_target_base = full_target[cols_base + ["tourney_id", "minutes"]].dropna(subset=cols_base).reset_index(drop=True)

    n_train, n_val, n_test = len(df_train_raw), len(df_val_raw), len(df_test_raw)
    assert len(full_target_base) == n_train + n_val + n_test, (
        f"Niespojnosc dlugosci {TARGET_YEAR}: {len(full_target_base)} vs {n_train + n_val + n_test}"
    )

    fatigue = compute_fatigue_for_2024(full_target_base)
    fat_train = fatigue.iloc[:n_train].reset_index(drop=True)
    fat_val = fatigue.iloc[n_train:n_train + n_val].reset_index(drop=True)
    fat_test = fatigue.iloc[n_train + n_val:].reset_index(drop=True)

    def attach(df_raw, fat):
        df_raw = df_raw.copy().reset_index(drop=True)
        for col in ("w_rest_days", "l_rest_days", "w_tourney_minutes", "l_tourney_minutes"):
            df_raw[col] = fat[col].to_numpy()
        return df_raw

    df_train_raw = attach(df_train_raw, fat_train)
    df_val_raw = attach(df_val_raw, fat_val)
    df_test_raw = attach(df_test_raw, fat_test)

    def build_split(df_raw, shuffle):
        sym = symmetrize_data(df_raw, shuffle=shuffle)
        raw_map = df_raw[["match_id", "w_rest_days", "l_rest_days",
                          "w_tourney_minutes", "l_tourney_minutes"]]
        sym = sym.merge(raw_map, on="match_id", how="left", validate="many_to_one")
        # w_/l_ -> p1_/p2_ wg perspektywy (y==1: p1=winner).
        is_p1_winner = (sym["y"] == 1).to_numpy()
        sym["p1_rest_days"] = np.where(is_p1_winner, sym["w_rest_days"], sym["l_rest_days"])
        sym["p2_rest_days"] = np.where(is_p1_winner, sym["l_rest_days"], sym["w_rest_days"])
        sym["p1_tourney_minutes"] = np.where(is_p1_winner, sym["w_tourney_minutes"], sym["l_tourney_minutes"])
        sym["p2_tourney_minutes"] = np.where(is_p1_winner, sym["l_tourney_minutes"], sym["w_tourney_minutes"])
        sym["rest_days_diff"] = sym["p1_rest_days"] - sym["p2_rest_days"]
        sym["tourney_minutes_diff"] = sym["p1_tourney_minutes"] - sym["p2_tourney_minutes"]
        return sym

    train_data = build_split(df_train_raw, shuffle=True)
    val_data = build_split(df_val_raw, shuffle=True)
    test_data = build_split(df_test_raw, shuffle=True)

    features = base_features + NEW_FEATURES
    X_train, y_train = train_data[features], train_data["y"]
    X_val, y_val = val_data[features], val_data["y"]
    X_test, y_test = test_data[features], test_data["y"]

    best_rf = RandomForestClassifier(
        **baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE
    )
    print(f"Liczba cech: {len(features)} (baseline: {len(base_features)}, nowe: {len(NEW_FEATURES)})")
    best_rf.fit(X_train, y_train)

    val_acc = float(accuracy_score(y_val, best_rf.predict(X_val)))
    test_acc = float(accuracy_score(y_test, best_rf.predict(X_test)))
    proba_test = best_rf.predict_proba(X_test)[:, 1]
    test_data["p1_win_probability"] = proba_test
    _, match_acc = compute_symmetric_match_evaluation(test_data)
    quality = evaluate_calibration_quality(y_test.to_numpy(), proba_test)

    importance = pd.DataFrame(
        {"feature": features, "importance": best_rf.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1

    print("\n" + "=" * 78)
    print("POROWNANIE: baseline RF vs baseline + FATIGUE")
    print("=" * 78)
    print(f"{'':<16} val      test     match    Brier")
    print(f"{'baseline':<16} {baseline_val_acc:.4f}   {baseline_test_acc:.4f}   {baseline_match_acc:.4f}")
    print(f"{'+ fatigue':<16} {val_acc:.4f}   {test_acc:.4f}   {match_acc:.4f}   {quality['brier_score']:.4f}")
    print("-" * 78)
    print(
        f"DELTA: val={val_acc - baseline_val_acc:+.4f}  "
        f"test={test_acc - baseline_test_acc:+.4f}  "
        f"match={match_acc - baseline_match_acc:+.4f}"
    )
    print()
    print("Pozycja nowych cech w waznosci:")
    for feat in NEW_FEATURES:
        row = importance[importance["feature"] == feat].iloc[0]
        print(f"  {feat:<22} rank {int(row['rank']):>2}/{len(features)}  importance={row['importance']:.4f}")
    print()
    print("UWAGA: CI dla match accuracy ~ +/-4 p.p. przy 590 meczach.")


if __name__ == "__main__":
    main()
