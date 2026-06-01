"""
Walidacja walk-forward zestawow cech na NOWYCH danych (2022-2025)
================================================================

Domyka luki i odswieza wszystkie wyniki cech na nowych danych w jednym, spojnym
biegu. Per rok (baseline cache'owany raz) testujemy 4 zestawy cech dodane do
baseline:
  surface  (3): court_pace_index + 2 interakcje serve x speed
  fatigue  (6): rest_days + tourney_minutes
  enriched (9): surface + fatigue
  elo      (4): elo_diff, surface_elo_diff, elo_win_prob, surface_elo_win_prob

Parujemy per-mecz z baseline -> delta pooled + McNemar. Te same tuned HP baseline
(ablation: zmieniamy tylko cechy).
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
from tennis_model_elo import build_elo_for_target_full, ELO_COLS

WORKDIR = Path(__file__).resolve().parent
BASE_DIR = WORKDIR.parent
DATA_DIR = BASE_DIR / "data" / "sample_data"
BASELINE_SCRIPT_PATH = (WORKDIR / "tennis_model.py").resolve()
TOUR = os.environ.get("TENNIS_TOUR", "atp")
HISTORY_START_YEAR = int(os.environ.get("TENNIS_HISTORY_START", "2001"))

_wf_env = os.environ.get("TENNIS_WF_YEARS")
TARGET_YEARS = ([int(y) for y in _wf_env.split(",")] if _wf_env
                else [2022, 2023, 2024, 2025])

SURFACE_FEATURES = ["court_pace_index", "ace_speed_diff", "first_won_speed_diff"]
FATIGUE_FEATURES = ["p1_rest_days", "p2_rest_days", "rest_days_diff",
                    "p1_tourney_minutes", "p2_tourney_minutes", "tourney_minutes_diff"]
ELO_FEATS = ["elo_diff", "surface_elo_diff", "elo_win_prob", "surface_elo_win_prob"]


def data_file(year):
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


_baseline_cache = None
_original_run_path = runpy.run_path


def _cached_run_path(p, *a, **k):
    global _baseline_cache
    try:
        resolved = Path(p).resolve()
    except (TypeError, OSError):
        resolved = None
    if resolved == BASELINE_SCRIPT_PATH:
        if _baseline_cache is None:
            _baseline_cache = _original_run_path(p, *a, **k)
        return _baseline_cache
    return _original_run_path(p, *a, **k)


runpy.run_path = _cached_run_path


def run_baseline(year):
    global _baseline_cache
    _baseline_cache = None
    os.environ["TENNIS_TARGET_YEAR"] = str(year)
    captured = io.StringIO()
    cwd = os.getcwd()
    os.chdir(WORKDIR)
    try:
        with contextlib.redirect_stdout(captured):
            return runpy.run_path(str(BASELINE_SCRIPT_PATH))
    finally:
        os.chdir(cwd)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    z = (abs(b - c) - 1) / math.sqrt(n)
    return z, math.erfc(abs(z) / math.sqrt(2))


def build_context(ns, year):
    """Buduje kontekst per match_id: court_pace + fatigue (w_/l_) + elo (w_/l_),
    wyrownany pozycyjnie do df_*_raw. Zwraca (full_target_base, ctx_df)."""
    cols_base = list(ns["cols_base"])
    history_files = [data_file(y) for y in range(HISTORY_START_YEAR, year)]

    full = pd.read_csv(data_file(year))
    full["tourney_date"] = pd.to_datetime(full["tourney_date"], format="%Y%m%d")
    full = full.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    full_base = full[cols_base + ["tourney_id", "minutes"]].dropna(subset=cols_base).reset_index(drop=True)

    # surface
    lookup = build_court_pace_lookup(history_files=history_files)
    cpi = np.array([court_pace_index(t, s, lookup)
                    for t, s in zip(full_base["tourney_id"], full_base["surface"])])
    # fatigue
    fat = compute_fatigue_for_2024(full_base, history_files=history_files)
    # elo (pelny rok -> dropna align)
    elo_full = build_elo_for_target_full(year)
    elo_aligned = elo_full.dropna(subset=cols_base).reset_index(drop=True)

    ctx = pd.DataFrame({
        "court_pace_index": cpi,
        "w_rest_days": fat["w_rest_days"].to_numpy(),
        "l_rest_days": fat["l_rest_days"].to_numpy(),
        "w_tourney_minutes": fat["w_tourney_minutes"].to_numpy(),
        "l_tourney_minutes": fat["l_tourney_minutes"].to_numpy(),
    })
    for col in ELO_COLS:
        ctx[col] = elo_aligned[col].to_numpy()
    return full_base, ctx


def eval_featureset(ns, ctx_cols_map, feature_list):
    """Trenuje RF (baseline HP) na baseline + feature_list, zwraca winner_perspective
    (match_id, correct_prediction) i match_accuracy."""
    symmetrize_data = ns["symmetrize_data"]
    compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
    base_features = list(ns["features"])
    baseline_search = ns["search"]
    RANDOM_STATE = ns["RANDOM_STATE"]

    n_tr, n_v, n_te = len(ns["df_train_raw"]), len(ns["df_val_raw"]), len(ns["df_test_raw"])
    ctx = ctx_cols_map["ctx"]
    ctx_tr = ctx.iloc[:n_tr].reset_index(drop=True)
    ctx_te = ctx.iloc[n_tr + n_v:].reset_index(drop=True)

    def attach(df_raw, c):
        df_raw = df_raw.copy().reset_index(drop=True)
        for col in ctx.columns:
            df_raw[col] = c[col].to_numpy()
        return df_raw

    df_train = attach(ns["df_train_raw"], ctx_tr)
    df_test = attach(ns["df_test_raw"], ctx_te)
    raw_cols = ["match_id"] + list(ctx.columns)

    def build(df_raw, shuffle):
        sym = symmetrize_data(df_raw, shuffle=shuffle)
        sym = sym.merge(df_raw[raw_cols], on="match_id", how="left", validate="many_to_one")
        is_p1w = (sym["y"] == 1).to_numpy()
        # surface interakcje
        sym["ace_speed_diff"] = (sym["p1_ace_rate"] - sym["p2_ace_rate"]) * sym["court_pace_index"]
        sym["first_won_speed_diff"] = (sym["p1_first_won_pct"] - sym["p2_first_won_pct"]) * sym["court_pace_index"]
        # fatigue
        sym["p1_rest_days"] = np.where(is_p1w, sym["w_rest_days"], sym["l_rest_days"])
        sym["p2_rest_days"] = np.where(is_p1w, sym["l_rest_days"], sym["w_rest_days"])
        sym["p1_tourney_minutes"] = np.where(is_p1w, sym["w_tourney_minutes"], sym["l_tourney_minutes"])
        sym["p2_tourney_minutes"] = np.where(is_p1w, sym["l_tourney_minutes"], sym["w_tourney_minutes"])
        sym["rest_days_diff"] = sym["p1_rest_days"] - sym["p2_rest_days"]
        sym["tourney_minutes_diff"] = sym["p1_tourney_minutes"] - sym["p2_tourney_minutes"]
        # elo
        p1e = np.where(is_p1w, sym["w_elo"], sym["l_elo"]); p2e = np.where(is_p1w, sym["l_elo"], sym["w_elo"])
        p1se = np.where(is_p1w, sym["w_surface_elo"], sym["l_surface_elo"])
        p2se = np.where(is_p1w, sym["l_surface_elo"], sym["w_surface_elo"])
        sym["elo_diff"] = p1e - p2e
        sym["surface_elo_diff"] = p1se - p2se
        sym["elo_win_prob"] = 1.0 / (1.0 + 10 ** ((p2e - p1e) / 400.0))
        sym["surface_elo_win_prob"] = 1.0 / (1.0 + 10 ** ((p2se - p1se) / 400.0))
        return sym

    train = build(df_train, True)
    test = build(df_test, True)
    feats = base_features + feature_list
    rf = RandomForestClassifier(**baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(train[feats], train["y"])
    test["p1_win_probability"] = rf.predict_proba(test[feats])[:, 1]
    ev, match = compute_symmetric_match_evaluation(test)
    return ev[["match_id", "correct_prediction"]], match


def main():
    SETS = {"surface": SURFACE_FEATURES, "fatigue": FATIGUE_FEATURES,
            "enriched": SURFACE_FEATURES + FATIGUE_FEATURES, "elo": ELO_FEATS}
    pairs = {k: [] for k in SETS}
    per_year = {k: [] for k in SETS}

    for year in TARGET_YEARS:
        print(f"\n===== ROK {year} =====", flush=True)
        ns = run_baseline(year)
        base_eval = ns["winner_perspective"][["match_id", "correct_prediction"]].copy()
        base_match = float(ns["match_accuracy"])
        print(f"  baseline match={base_match:.4f}", flush=True)
        _, ctx = build_context(ns, year)
        for name, feats in SETS.items():
            ev, match = eval_featureset(ns, {"ctx": ctx}, feats)
            merged = base_eval.merge(ev, on="match_id", suffixes=("_base", "_var"))
            for _, r in merged.iterrows():
                pairs[name].append((bool(r["correct_prediction_base"]), bool(r["correct_prediction_var"])))
            per_year[name].append({"year": year, "baseline": base_match, "variant": match, "delta": match - base_match})
            print(f"    {name:<10} match={match:.4f}  delta={match-base_match:+.4f}", flush=True)

    os.environ.pop("TENNIS_TARGET_YEAR", None)

    print("\n" + "=" * 74)
    print("WALK-FORWARD CECH (nowe dane 2022-2025) vs baseline")
    print("=" * 74)
    for name in SETS:
        df = pd.DataFrame(per_year[name])
        arr = np.array(pairs[name]); bc, vc = arr[:, 0], arr[:, 1]
        b = int(np.sum(bc & ~vc)); c = int(np.sum(~bc & vc))
        z, p = mcnemar(b, c)
        pos = int((df["delta"] > 0).sum())
        sig = "ISTOTNE" if (p < 0.05 and c > b) else ("ISTOTNE-na-niekorzysc" if p < 0.05 else "brak istotnosci")
        print(f"\n--- {name} ({len(arr[0]) if False else len(SETS[name])} cech) ---")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  POOLED ({len(arr)}): baseline={bc.mean():.4f}  {name}={vc.mean():.4f}  "
              f"delta={vc.mean()-bc.mean():+.4f}  (dodatnie {pos}/{len(df)} lat)")
        print(f"  McNemar: b={b} c={c} z={z:.2f} p={p:.4f} -> {sig}")


if __name__ == "__main__":
    main()
