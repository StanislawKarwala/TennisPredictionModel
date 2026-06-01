"""
Sprint 6: Wielo-sezonowy trening + uczciwy test boostingu (RF vs HGB vs XGBoost)
===============================================================================

Dlaczego: dotychczasowa architektura trenowala TYLKO na roku docelowym (~3500
probek). Sprint 2 pokazal, ze HGB nie bije RF -- ale na tak malej probie boosting
nie ma jak rozwinac przewagi. Tu zmieniamy architekture: trenujemy na WIELU
sezonach (2010-2023, ~79k probek), walidujemy na 2024, testujemy na 2025. To
wlasciwy test hipotezy "wiecej danych => boosting oplacalny".

Reuzywamy funkcje feature-engineering z baseline (przez namespace), wiec cechy sa
IDENTYCZNE (40 cech) -- jedyne zmienne to ilosc danych treningowych i algorytm.

Konfiguracja przez env:
  TENNIS_TRAIN_START (domyslnie 2010), TENNIS_WARMUP_START (2001),
  TENNIS_VAL_YEAR (2024), TENNIS_TEST_YEAR (2025).
"""

from __future__ import annotations

import contextlib
import io
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


BASE_SCRIPT = Path(__file__).with_name("main_48_cech.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
TOUR = os.environ.get("TENNIS_TOUR", "atp")

WARMUP_START = int(os.environ.get("TENNIS_WARMUP_START", "2001"))
TRAIN_START = int(os.environ.get("TENNIS_TRAIN_START", "2010"))
VAL_YEAR = int(os.environ.get("TENNIS_VAL_YEAR", "2024"))
TEST_YEAR = int(os.environ.get("TENNIS_TEST_YEAR", "2025"))
RANDOM_STATE = 42


def data_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


def run_baseline_quietly() -> dict:
    """Uruchamia baseline raz, by pobrac funkcje feature-engineering i ewaluacji."""
    original_cwd = os.getcwd()
    captured = io.StringIO()
    os.chdir(BASE_SCRIPT.parent)
    try:
        with contextlib.redirect_stdout(captured):
            return runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)


def load_years(years, cols_base):
    # WAZNE: 'season' = rok PLIKU, nie tourney_date.dt.year. Plik sezonu 2025
    # zaczyna sie od United Cup z konca grudnia 2024, wiec data myli sezon.
    parts = []
    for y in years:
        df = pd.read_csv(data_file(y))
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
        df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
        sub = df[cols_base + ["match_num"]].dropna(subset=cols_base).copy()
        sub["season"] = y
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)


def add_static_features(df, ROUND_ORDER):
    df = df.copy()
    df["winner_rank_log"] = np.log(df["winner_rank"])
    df["loser_rank_log"] = np.log(df["loser_rank"])
    df["winner_rank_pts_log"] = np.log(df["winner_rank_points"])
    df["loser_rank_pts_log"] = np.log(df["loser_rank_points"])
    df["winner_is_lefty"] = (df["winner_hand"] == "L").astype(int)
    df["loser_is_lefty"] = (df["loser_hand"] == "L").astype(int)
    df["round_encoded"] = df["round"].map(ROUND_ORDER).fillna(3)
    return df


def tune_and_eval(name, estimator, param_dist, n_iter,
                  X_tr_cv, y_tr_cv, X_tr_fit, y_tr_fit,
                  X_val, y_val, val_data, X_test, y_test, test_data,
                  compute_symmetric_match_evaluation, evaluate_calibration_quality):
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        estimator, param_dist, n_iter=n_iter, cv=tscv,
        scoring="neg_log_loss", n_jobs=-1, random_state=RANDOM_STATE,
    )
    search.fit(X_tr_cv, y_tr_cv)
    best = search.best_estimator_
    best.fit(X_tr_fit, y_tr_fit)

    val_acc = float(accuracy_score(y_val, best.predict(X_val)))
    test_acc = float(accuracy_score(y_test, best.predict(X_test)))

    vd = val_data.copy()
    vd["p1_win_probability"] = best.predict_proba(X_val)[:, 1]
    _, val_match = compute_symmetric_match_evaluation(vd)
    td = test_data.copy()
    td["p1_win_probability"] = best.predict_proba(X_test)[:, 1]
    _, test_match = compute_symmetric_match_evaluation(td)
    q = evaluate_calibration_quality(y_test.to_numpy(), best.predict_proba(X_test)[:, 1])
    return {
        "name": name, "val_acc": val_acc, "test_acc": test_acc,
        "val_match": val_match, "test_match": test_match,
        "brier": q["brier_score"], "logloss": q["log_loss"], "ece": q["expected_calibration_error"],
        "best_params": search.best_params_,
    }


def main():
    print("Uruchamiam baseline raz (pobranie funkcji feature-engineering)...", flush=True)
    ns = run_baseline_quietly()
    add_dynamic_features = ns["add_dynamic_features"]
    symmetrize_data = ns["symmetrize_data"]
    compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
    evaluate_calibration_quality = ns["evaluate_calibration_quality"]
    features = list(ns["features"])
    cols_base = list(ns["cols_base"])
    ROUND_ORDER = ns["ROUND_ORDER"]

    print(f"Trening {TRAIN_START}-{VAL_YEAR-1} | walidacja {VAL_YEAR} | test {TEST_YEAR}", flush=True)
    warmup_desc = f"{WARMUP_START}-{TRAIN_START-1}" if TRAIN_START > WARMUP_START else "BRAK (trening od najwczesniejszego sezonu)"
    print(f"Warmup cech: {warmup_desc}", flush=True)

    # Jedno wczytanie WARMUP_START..TEST_YEAR, potem podzial po sezonie.
    # historical = sezony przed TRAIN_START (tylko rozgrzewka cech, moze byc puste);
    # span = TRAIN_START..TEST_YEAR (cechy liczone). Slice zachowuje kolumny nawet
    # gdy historical ma 0 wierszy -> add_dynamic_features radzi sobie z pusta historia.
    full = load_years(range(WARMUP_START, TEST_YEAR + 1), cols_base)
    full = add_static_features(full, ROUND_ORDER)
    historical = full[full["season"] < TRAIN_START].reset_index(drop=True)
    span = full[full["season"] >= TRAIN_START].reset_index(drop=True)

    print(f"Licze cechy dynamiczne dla {len(span)} meczow ({TRAIN_START}-{TEST_YEAR})"
          f" (rozgrzewka: {len(historical)} meczow)...", flush=True)
    span_feat = add_dynamic_features(span, historical)

    # Label encoding -- fit TYLKO na treningu (sezon < VAL_YEAR), bez wgladu w val/test.
    train_mask = span_feat["season"] < VAL_YEAR
    le_surface, le_level = LabelEncoder(), LabelEncoder()
    le_surface.fit(span_feat.loc[train_mask, "surface"])
    le_level.fit(span_feat.loc[train_mask, "tourney_level"].astype(str))

    def safe_transform(le, series):
        known = set(le.classes_)
        s = series.astype(str).where(series.astype(str).isin(known), le.classes_[0])
        return le.transform(s)

    span_feat["surface_encoded"] = safe_transform(le_surface, span_feat["surface"])
    span_feat["tourney_level_encoded"] = safe_transform(le_level, span_feat["tourney_level"].astype(str))

    train_raw = span_feat[span_feat["season"] < VAL_YEAR].reset_index(drop=True)
    val_raw = span_feat[span_feat["season"] == VAL_YEAR].reset_index(drop=True)
    test_raw = span_feat[span_feat["season"] == TEST_YEAR].reset_index(drop=True)
    for frame in (train_raw, val_raw, test_raw):
        frame["match_id"] = range(len(frame))

    print(f"Mecze: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}", flush=True)

    # Symetryzacja: ordered (do CV chronologicznego) i shuffled (do finalnego fitu).
    train_cv = symmetrize_data(train_raw, shuffle=False)
    train_fit = symmetrize_data(train_raw, shuffle=True)
    val_data = symmetrize_data(val_raw, shuffle=True)
    test_data = symmetrize_data(test_raw, shuffle=True)

    X_tr_cv, y_tr_cv = train_cv[features], train_cv["y"]
    X_tr_fit, y_tr_fit = train_fit[features], train_fit["y"]
    X_val, y_val = val_data[features], val_data["y"]
    X_test, y_test = test_data[features], test_data["y"]
    print(f"Probki treningowe (po symetryzacji): {len(X_tr_fit)}", flush=True)

    results = []

    print("\n[1/3] Random Forest...", flush=True)
    results.append(tune_and_eval(
        "RandomForest",
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [10, 20, None],
         "min_samples_leaf": [2, 5, 10], "max_features": ["sqrt", "log2"],
         "max_samples": [0.8, 1.0]},
        8, X_tr_cv, y_tr_cv, X_tr_fit, y_tr_fit, X_val, y_val, val_data,
        X_test, y_test, test_data, compute_symmetric_match_evaluation, evaluate_calibration_quality))

    print("[2/3] HistGradientBoosting...", flush=True)
    results.append(tune_and_eval(
        "HistGradBoost",
        HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=False),
        {"learning_rate": [0.03, 0.05, 0.1], "max_iter": [300, 500, 800],
         "max_leaf_nodes": [31, 63], "min_samples_leaf": [20, 50, 100],
         "l2_regularization": [0.0, 0.1, 1.0], "max_features": [0.6, 0.8, 1.0]},
        12, X_tr_cv, y_tr_cv, X_tr_fit, y_tr_fit, X_val, y_val, val_data,
        X_test, y_test, test_data, compute_symmetric_match_evaluation, evaluate_calibration_quality))

    if HAS_XGB:
        print("[3/3] XGBoost...", flush=True)
        results.append(tune_and_eval(
            "XGBoost",
            XGBClassifier(tree_method="hist", objective="binary:logistic",
                          eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
            {"n_estimators": [300, 500, 800], "max_depth": [4, 6, 8],
             "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.9],
             "colsample_bytree": [0.7, 0.9], "min_child_weight": [1, 5, 10]},
            12, X_tr_cv, y_tr_cv, X_tr_fit, y_tr_fit, X_val, y_val, val_data,
            X_test, y_test, test_data, compute_symmetric_match_evaluation, evaluate_calibration_quality))
    else:
        print("[3/3] XGBoost POMINIETY (brak biblioteki).", flush=True)

    print("\n" + "=" * 88)
    print(f"WIELO-SEZONOWY TRENING ({TRAIN_START}-{VAL_YEAR-1}, ~{len(X_tr_fit)} probek) | test {TEST_YEAR}")
    print("=" * 88)
    print(f"{'model':<16}{'val_match':>11}{'test_match':>12}{'Brier':>9}{'logloss':>9}{'ECE':>8}")
    for r in results:
        print(f"{r['name']:<16}{r['val_match']:>11.4f}{r['test_match']:>12.4f}"
              f"{r['brier']:>9.4f}{r['logloss']:>9.4f}{r['ece']:>8.4f}")
    print("-" * 88)
    rf = next(r for r in results if r["name"] == "RandomForest")
    for r in results:
        if r["name"] != "RandomForest":
            print(f"DELTA ({r['name']} - RF): test_match={r['test_match']-rf['test_match']:+.4f}  "
                  f"Brier={r['brier']-rf['brier']:+.4f}")
    print()
    print("Najlepsze hiperparametry:")
    for r in results:
        print(f"  {r['name']}: {r['best_params']}")
    print()
    print(f"UWAGA: test = caly sezon {TEST_YEAR} (~{len(test_raw)} meczow). CI ~ +/-2 p.p.")


if __name__ == "__main__":
    main()
