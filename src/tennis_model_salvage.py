"""
Analiza SALVAGE (Sprint 5): czy z walk-forward da sie uratowac robust sygnal?
=============================================================================

Testuje cztery WARIANTY wzbogacenia na IDENTYCZNYCH meczach testowych
(parowanie per-mecz) dla kazdego roku 2021-2024:

  full     = baseline + 9 cech (3 speed + 6 fatigue)            [stan z walk-forward]
  speed3   = baseline + court_pace_index + ace_speed_diff + first_won_speed_diff
  narrow2  = baseline + first_won_speed_diff + ace_speed_diff   [tylko interakcje]
  single1  = baseline + first_won_speed_diff                    [najsilniejsza cecha]

Dla kazdego wariantu liczymy: delta per rok, delta pooled, McNemar (parowany),
oraz korelacje delta vs sila baseline (hipoteza "pomaga gdy baseline slaby").

Reuzywa machinerii z tennis_model_walkforward (te same splity, te same tuned HP).
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tennis_model_surface_speed import build_court_pace_lookup, court_pace_index
from tennis_model_fatigue import compute_fatigue_for_2024


BASE_SCRIPT = Path(__file__).with_name("tennis_model.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
TOUR = os.environ.get("TENNIS_TOUR", "atp")
HISTORY_START_YEAR = int(os.environ.get("TENNIS_HISTORY_START", "2001"))


def data_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


_wf_env = os.environ.get("TENNIS_WF_YEARS")
TARGET_YEARS = ([int(y) for y in _wf_env.split(",")] if _wf_env
                else [2020, 2021, 2022, 2023, 2024, 2025])

SPEED_FEATURES = ["court_pace_index", "ace_speed_diff", "first_won_speed_diff"]
FATIGUE_FEATURES = [
    "p1_rest_days", "p2_rest_days", "rest_days_diff",
    "p1_tourney_minutes", "p2_tourney_minutes", "tourney_minutes_diff",
]

VARIANTS = {
    "full": SPEED_FEATURES + FATIGUE_FEATURES,
    "speed3": SPEED_FEATURES,
    "narrow2": ["first_won_speed_diff", "ace_speed_diff"],
    "single1": ["first_won_speed_diff"],
}


def run_baseline_for_year(year: int) -> dict:
    os.environ["TENNIS_TARGET_YEAR"] = str(year)
    original_cwd = os.getcwd()
    captured = io.StringIO()
    os.chdir(BASE_SCRIPT.parent)
    try:
        with contextlib.redirect_stdout(captured):
            return runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)


def build_enriched_splits(ns: dict, year: int):
    """Zwraca (train_data, test_data, base_features, search, RANDOM_STATE,
    base_eval, base_match). train_data/test_data zawieraja WSZYSTKIE nowe cechy,
    wiec kazdy wariant to po prostu inny podzbior kolumn na tych samych ramkach."""
    symmetrize_data = ns["symmetrize_data"]
    compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
    baseline_search = ns["search"]
    RANDOM_STATE = ns["RANDOM_STATE"]
    base_features = list(ns["features"])
    cols_base = list(ns["cols_base"])

    df_train_raw = ns["df_train_raw"].copy()
    df_val_raw = ns["df_val_raw"].copy()
    df_test_raw = ns["df_test_raw"].copy()
    n_train, n_val, n_test = len(df_train_raw), len(df_val_raw), len(df_test_raw)

    history_files = [data_file(y) for y in range(HISTORY_START_YEAR, year)]

    full = pd.read_csv(data_file(year))
    full["tourney_date"] = pd.to_datetime(full["tourney_date"], format="%Y%m%d")
    full = full.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    full_base = full[cols_base + ["tourney_id", "minutes"]].dropna(subset=cols_base).reset_index(drop=True)
    assert len(full_base) == n_train + n_val + n_test

    lookup = build_court_pace_lookup(history_files=history_files)
    cpi = np.array([court_pace_index(t, s, lookup)
                    for t, s in zip(full_base["tourney_id"], full_base["surface"])])
    fatigue = compute_fatigue_for_2024(full_base, history_files=history_files)

    context = pd.DataFrame({
        "court_pace_index": cpi,
        "w_rest_days": fatigue["w_rest_days"].to_numpy(),
        "l_rest_days": fatigue["l_rest_days"].to_numpy(),
        "w_tourney_minutes": fatigue["w_tourney_minutes"].to_numpy(),
        "l_tourney_minutes": fatigue["l_tourney_minutes"].to_numpy(),
    })
    ctx_train = context.iloc[:n_train].reset_index(drop=True)
    ctx_test = context.iloc[n_train + n_val:].reset_index(drop=True)

    def attach(df_raw, ctx):
        df_raw = df_raw.copy().reset_index(drop=True)
        for col in context.columns:
            df_raw[col] = ctx[col].to_numpy()
        return df_raw

    df_train_raw = attach(df_train_raw, ctx_train)
    df_test_raw = attach(df_test_raw, ctx_test)
    raw_ctx_cols = ["match_id"] + list(context.columns)

    def build_split(df_raw, shuffle):
        sym = symmetrize_data(df_raw, shuffle=shuffle)
        sym = sym.merge(df_raw[raw_ctx_cols], on="match_id", how="left", validate="many_to_one")
        is_p1_winner = (sym["y"] == 1).to_numpy()
        sym["ace_speed_diff"] = (sym["p1_ace_rate"] - sym["p2_ace_rate"]) * sym["court_pace_index"]
        sym["first_won_speed_diff"] = (sym["p1_first_won_pct"] - sym["p2_first_won_pct"]) * sym["court_pace_index"]
        sym["p1_rest_days"] = np.where(is_p1_winner, sym["w_rest_days"], sym["l_rest_days"])
        sym["p2_rest_days"] = np.where(is_p1_winner, sym["l_rest_days"], sym["w_rest_days"])
        sym["p1_tourney_minutes"] = np.where(is_p1_winner, sym["w_tourney_minutes"], sym["l_tourney_minutes"])
        sym["p2_tourney_minutes"] = np.where(is_p1_winner, sym["l_tourney_minutes"], sym["w_tourney_minutes"])
        sym["rest_days_diff"] = sym["p1_rest_days"] - sym["p2_rest_days"]
        sym["tourney_minutes_diff"] = sym["p1_tourney_minutes"] - sym["p2_tourney_minutes"]
        return sym

    train_data = build_split(df_train_raw, shuffle=True)
    test_data = build_split(df_test_raw, shuffle=True)

    base_eval = ns["winner_perspective"][["match_id", "correct_prediction"]].copy()
    base_match = float(ns["match_accuracy"])

    return (train_data, test_data, base_features, baseline_search, RANDOM_STATE,
            compute_symmetric_match_evaluation, base_eval, base_match)


def eval_variant(train_data, test_data, base_features, new_feats, search,
                 RANDOM_STATE, compute_eval):
    features = base_features + new_feats
    rf = RandomForestClassifier(**search.best_params_, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(train_data[features], train_data["y"])
    test_data = test_data.copy()
    test_data["p1_win_probability"] = rf.predict_proba(test_data[features])[:, 1]
    ev, match = compute_eval(test_data)
    return ev[["match_id", "correct_prediction"]], match


def mcnemar(b: int, c: int) -> tuple[float, float]:
    n = b + c
    if n == 0:
        return 0.0, 1.0
    # Korekta ciaglosci nie moze zejsc ponizej zera -- dla b==c byloby z<0,
    # a abs(z) dawalby p<1 zamiast poprawnego p=1 (brak roznicy).
    z = max(abs(b - c) - 1, 0) / math.sqrt(n)
    p = math.erfc(z / math.sqrt(2))
    return z, p


def main() -> None:
    # per_year_deltas[variant] = list of (year, base_match, var_match, delta)
    per_year = {v: [] for v in VARIANTS}
    # pooled pairs[variant] = list of (base_correct, var_correct)
    pooled = {v: [] for v in VARIANTS}
    base_by_year = {}

    for year in TARGET_YEARS:
        print(f"\n===== ROK {year} =====", flush=True)
        ns = run_baseline_for_year(year)
        (train_data, test_data, base_features, search, RS, compute_eval,
         base_eval, base_match) = build_enriched_splits(ns, year)
        base_by_year[year] = base_match
        print(f"  baseline={base_match:.4f}  (n_test_meczow={len(base_eval)})", flush=True)

        for vname, feats in VARIANTS.items():
            var_eval, var_match = eval_variant(
                train_data, test_data, base_features, feats, search, RS, compute_eval
            )
            merged = base_eval.merge(var_eval, on="match_id", suffixes=("_base", "_var"))
            per_year[vname].append((year, base_match, var_match, var_match - base_match))
            for _, r in merged.iterrows():
                pooled[vname].append((bool(r["correct_prediction_base"]),
                                      bool(r["correct_prediction_var"])))
            print(f"    {vname:8s} ({len(feats)} cech): {var_match:.4f}  "
                  f"delta={var_match-base_match:+.4f}", flush=True)

    os.environ.pop("TENNIS_TARGET_YEAR", None)

    print("\n" + "=" * 78)
    print("PODSUMOWANIE SALVAGE")
    print("=" * 78)
    for vname in VARIANTS:
        rows = per_year[vname]
        pairs = np.array(pooled[vname])
        bc = pairs[:, 0]
        vc = pairs[:, 1]
        N = len(pairs)
        pooled_delta = float(vc.mean() - bc.mean())
        b = int(np.sum(bc & ~vc))
        c = int(np.sum(~bc & vc))
        z, pval = mcnemar(b, c)
        deltas = [d for (_, _, _, d) in rows]
        bases = [bb for (_, bb, _, _) in rows]
        pos_years = sum(1 for d in deltas if d > 0)
        # korelacja delta vs sila baseline (hipoteza a)
        if len(set(bases)) > 1:
            corr = float(np.corrcoef(bases, deltas)[0, 1])
        else:
            corr = float("nan")
        print(f"\n--- WARIANT: {vname} ({len(VARIANTS[vname])} nowych cech) ---")
        for (year, bm, vm, d) in rows:
            print(f"   {year}: base={bm:.4f} var={vm:.4f} delta={d:+.4f}")
        print(f"   POOLED: delta={pooled_delta:+.4f} (N={N})  "
              f"dodatnie {pos_years}/{len(rows)} lat")
        print(f"   McNemar: b={b} c={c} z={z:.2f} p={pval:.4f}")
        print(f"   corr(baseline_acc, delta) = {corr:+.3f}  "
              f"(ujemna => pomaga gdy baseline slaby)")


if __name__ == "__main__":
    main()
