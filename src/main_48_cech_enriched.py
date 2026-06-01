"""
Eksperyment Sprint 3d: Model zbiorczy (surface_speed + fatigue)
===============================================================

Laczy dwa wygrywajace zestawy cech ze Sprint 3:
  - surface_speed (court_pace_index + interakcje serve x speed)  -> +1.69 p.p.
  - fatigue (rest_days + tourney_minutes)                        -> +1.36 p.p.

Pytanie: czy zyski sie SUMUJA, czy uderza curse of dimensionality (jak w
wariancie sliceaware ze Sprint 1, gdzie 33 cechy naraz daly gorzej)?

Reuzywa buildery cech z modulow eksperymentalnych (bez duplikacji kodu).
Ablation: te same tuned HP baseline, metryka symetryczna z Sprint 1.
"""

from __future__ import annotations

import contextlib
import io
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from main_48_cech_surface_speed import build_court_pace_lookup, court_pace_index
from main_48_cech_fatigue import compute_fatigue_for_2024


BASE_SCRIPT = Path(__file__).with_name("main_48_cech.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
TOUR = os.environ.get("TENNIS_TOUR", "atp")
TARGET_YEAR = int(os.environ.get("TENNIS_TARGET_YEAR", "2025"))
HISTORY_START_YEAR = int(os.environ.get("TENNIS_HISTORY_START", "2001"))


def data_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


# is_indoor pominiete (rank 44/44 bezuzyteczne + brak kolumny w atp_matches_*).
SPEED_FEATURES = ["court_pace_index", "ace_speed_diff", "first_won_speed_diff"]
FATIGUE_FEATURES = [
    "p1_rest_days", "p2_rest_days", "rest_days_diff",
    "p1_tourney_minutes", "p2_tourney_minutes", "tourney_minutes_diff",
]
NEW_FEATURES = SPEED_FEATURES + FATIGUE_FEATURES


def execute_base_pipeline_quietly() -> dict:
    original_cwd = os.getcwd()
    captured = io.StringIO()
    os.chdir(BASE_SCRIPT.parent)
    try:
        with contextlib.redirect_stdout(captured):
            return runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)


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

    # Wspolny "kontekst" roku docelowego (tourney_id, surface, minutes), te same wiersze co baseline.
    full_target = pd.read_csv(data_file(TARGET_YEAR))
    full_target["tourney_date"] = pd.to_datetime(full_target["tourney_date"], format="%Y%m%d")
    full_target = full_target.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    full_target_base = full_target[cols_base + ["tourney_id", "minutes"]].dropna(subset=cols_base).reset_index(drop=True)

    n_train, n_val, n_test = len(df_train_raw), len(df_val_raw), len(df_test_raw)
    assert len(full_target_base) == n_train + n_val + n_test

    # --- surface speed ---
    print("Buduje court_pace_index...")
    lookup = build_court_pace_lookup()
    cpi = np.array([court_pace_index(t, s, lookup)
                    for t, s in zip(full_target_base["tourney_id"], full_target_base["surface"])])

    # --- fatigue ---
    print("Licze cechy zmeczenia...")
    fatigue = compute_fatigue_for_2024(full_target_base)

    # Sklej kontekst per match (wyrownany 1:1 do full_2024_base).
    context = pd.DataFrame({
        "court_pace_index": cpi,
        "w_rest_days": fatigue["w_rest_days"].to_numpy(),
        "l_rest_days": fatigue["l_rest_days"].to_numpy(),
        "w_tourney_minutes": fatigue["w_tourney_minutes"].to_numpy(),
        "l_tourney_minutes": fatigue["l_tourney_minutes"].to_numpy(),
    })
    ctx_train = context.iloc[:n_train].reset_index(drop=True)
    ctx_val = context.iloc[n_train:n_train + n_val].reset_index(drop=True)
    ctx_test = context.iloc[n_train + n_val:].reset_index(drop=True)

    def attach(df_raw, ctx):
        df_raw = df_raw.copy().reset_index(drop=True)
        for col in context.columns:
            df_raw[col] = ctx[col].to_numpy()
        return df_raw

    df_train_raw = attach(df_train_raw, ctx_train)
    df_val_raw = attach(df_val_raw, ctx_val)
    df_test_raw = attach(df_test_raw, ctx_test)

    raw_ctx_cols = ["match_id"] + list(context.columns)

    def build_split(df_raw, shuffle):
        sym = symmetrize_data(df_raw, shuffle=shuffle)
        sym = sym.merge(df_raw[raw_ctx_cols], on="match_id", how="left", validate="many_to_one")
        is_p1_winner = (sym["y"] == 1).to_numpy()
        # surface speed interakcje (court_pace_index symetryczny kontekst)
        sym["ace_speed_diff"] = (sym["p1_ace_rate"] - sym["p2_ace_rate"]) * sym["court_pace_index"]
        sym["first_won_speed_diff"] = (sym["p1_first_won_pct"] - sym["p2_first_won_pct"]) * sym["court_pace_index"]
        # fatigue w_/l_ -> p1_/p2_
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

    print("\n" + "=" * 80)
    print("MODEL ZBIORCZY: baseline + surface_speed + fatigue")
    print("=" * 80)
    print(f"{'':<22} val      test     match    Brier")
    print(f"{'baseline':<22} {baseline_val_acc:.4f}   {baseline_test_acc:.4f}   {baseline_match_acc:.4f}")
    print(f"{'+ speed + fatigue':<22} {val_acc:.4f}   {test_acc:.4f}   {match_acc:.4f}   {quality['brier_score']:.4f}")
    print("-" * 80)
    print(
        f"DELTA: val={val_acc - baseline_val_acc:+.4f}  "
        f"test={test_acc - baseline_test_acc:+.4f}  "
        f"match={match_acc - baseline_match_acc:+.4f}"
    )
    print()
    print("Przypomnienie (osobno): surface_speed match +0.0169 | fatigue match +0.0136")
    print()
    print("Top 5 nowych cech wg waznosci:")
    new_imp = importance[importance["feature"].isin(NEW_FEATURES)].head(5)
    for _, row in new_imp.iterrows():
        print(f"  {row['feature']:<22} rank {int(row['rank']):>2}/{len(features)}  importance={row['importance']:.4f}")
    print()
    print("UWAGA: CI dla match accuracy ~ +/-4 p.p. przy 590 meczach -- walidacja w Sprint 4.")


if __name__ == "__main__":
    main()
