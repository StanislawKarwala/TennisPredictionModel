"""
Sprint 4: Walidacja walk-forward (wiele lat) + test istotnosci
==============================================================

Problem: pojedynczy test set 2024 (~590 meczow) ma CI ~ +/-4 p.p., wiec zyski
+1-2 p.p. ze Sprint 3 sa w szumie. Walk-forward daje WIELE niezaleznych test
setow -> wezsze CI + sprawdzenie spojnosci delty miedzy latami.

Dla kazdego roku docelowego Y in {2021,2022,2023,2024}:
  - historia = sezony 2018..Y-1,
  - baseline (RF, 40 cech) trenowany na 60% Y, testowany na 20% Y,
  - model wzbogacony (baseline + surface_speed + fatigue) na tych samych danych,
  - zbieramy per-mecz poprawnosc OBU modeli (te same mecze) -> test parowany.

Raport: delta per rok, delta pooled (wszystkie mecze razem) z CI, oraz McNemar
(parowany test istotnosci na meczach, gdzie modele sie roznia).

Reuzywa parametryzowany baseline (TENNIS_TARGET_YEAR) i buildery cech ze Sprint 3.
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
from sklearn.metrics import accuracy_score

from main_48_cech_surface_speed import build_court_pace_lookup, court_pace_index
from main_48_cech_fatigue import compute_fatigue_for_2024


BASE_SCRIPT = Path(__file__).with_name("main_48_cech.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
TOUR = os.environ.get("TENNIS_TOUR", "atp")
HISTORY_START_YEAR = int(os.environ.get("TENNIS_HISTORY_START", "2001"))


def data_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


# Wiecej danych (2001+) => wiecej sezonow testowych => ciasniejsze CI.
# Konfigurowalne przez env TENNIS_WF_YEARS="2020,2021,...".
_wf_env = os.environ.get("TENNIS_WF_YEARS")
TARGET_YEARS = ([int(y) for y in _wf_env.split(",")] if _wf_env
                else [2020, 2021, 2022, 2023, 2024, 2025])

SPEED_FEATURES = ["court_pace_index", "ace_speed_diff", "first_won_speed_diff"]
FATIGUE_FEATURES = [
    "p1_rest_days", "p2_rest_days", "rest_days_diff",
    "p1_tourney_minutes", "p2_tourney_minutes", "tourney_minutes_diff",
]
NEW_FEATURES = SPEED_FEATURES + FATIGUE_FEATURES


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


def enriched_eval_for_year(ns: dict, year: int):
    """Buduje model wzbogacony dla danego roku, zwraca (baseline_eval,
    enriched_eval, baseline_match_acc, enriched_match_acc).
    *_eval to ramki z kolumnami match_id, correct_prediction (na test secie)."""
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
    features = base_features + NEW_FEATURES

    enr_rf = RandomForestClassifier(
        **baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE
    )
    enr_rf.fit(train_data[features], train_data["y"])
    test_data["p1_win_probability"] = enr_rf.predict_proba(test_data[features])[:, 1]
    enr_eval, enr_match = compute_symmetric_match_evaluation(test_data)

    # baseline eval (z namespace -- juz policzony winner_perspective)
    base_eval = ns["winner_perspective"][["match_id", "correct_prediction"]].copy()
    base_match = float(ns["match_accuracy"])

    return base_eval, enr_eval[["match_id", "correct_prediction"]], base_match, enr_match


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Parowany test McNemar (normal approx z poprawka ciaglosci).
    b = baseline poprawny & enriched bledny; c = odwrotnie. Zwraca (z, p_two_sided)."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    z = (abs(b - c) - 1) / math.sqrt(n) if n > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def wilson_delta_ci(p: float, n: int, z: float = 1.96) -> float:
    """Polowa szerokosci ~95% CI dla proporcji (do orientacji co do szumu)."""
    if n == 0:
        return float("nan")
    return z * math.sqrt(p * (1 - p) / n)


def main() -> None:
    rows = []
    all_pairs = []  # (base_correct, enr_correct) po wszystkich latach
    for year in TARGET_YEARS:
        print(f"\n===== ROK DOCELOWY {year} =====", flush=True)
        ns = run_baseline_for_year(year)
        base_eval, enr_eval, base_match, enr_match = enriched_eval_for_year(ns, year)
        merged = base_eval.merge(enr_eval, on="match_id", suffixes=("_base", "_enr"))
        n_test = len(merged)
        rows.append({
            "year": year, "n_test": n_test,
            "baseline": base_match, "enriched": enr_match,
            "delta": enr_match - base_match,
        })
        for _, r in merged.iterrows():
            all_pairs.append((bool(r["correct_prediction_base"]), bool(r["correct_prediction_enr"])))
        print(f"  baseline={base_match:.4f}  enriched={enr_match:.4f}  delta={enr_match-base_match:+.4f}  (n={n_test})", flush=True)

    os.environ.pop("TENNIS_TARGET_YEAR", None)

    df = pd.DataFrame(rows)
    pairs = np.array(all_pairs)
    base_correct = pairs[:, 0]
    enr_correct = pairs[:, 1]
    N = len(pairs)
    pooled_base = float(base_correct.mean())
    pooled_enr = float(enr_correct.mean())
    pooled_delta = pooled_enr - pooled_base

    # McNemar na zsumowanych meczach
    b = int(np.sum(base_correct & ~enr_correct))
    c = int(np.sum(~base_correct & enr_correct))
    z, pval = mcnemar(b, c)

    print("\n" + "=" * 72)
    print("WALK-FORWARD: baseline vs enriched (surface_speed + fatigue)")
    print("=" * 72)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 72)
    print(f"POOLED ({N} meczow): baseline={pooled_base:.4f}  enriched={pooled_enr:.4f}  "
          f"delta={pooled_delta:+.4f}")
    print(f"  Orientacyjne 95% CI (pojedyncza proporcja): +/-{wilson_delta_ci(pooled_enr, N):.4f}")
    print(f"  Spojnosc: delta dodatnia w {int((df['delta']>0).sum())}/{len(df)} latach")
    print()
    print(f"McNemar (parowany): b(base+,enr-)={b}  c(base-,enr+)={c}  z={z:.2f}  p(two-sided)={pval:.4f}")
    if pval < 0.05 and c > b:
        print("  => Roznica ISTOTNA statystycznie (p<0.05) na korzysc enriched.")
    elif pval < 0.05 and b > c:
        print("  => Roznica istotna, ale na NIEKORZYSC enriched.")
    else:
        print("  => Brak istotnosci (p>=0.05): zysk w granicach szumu.")


if __name__ == "__main__":
    main()
