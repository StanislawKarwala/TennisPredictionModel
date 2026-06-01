"""
Sprint 5: Surface-adjusted Elo jako cecha (z walidacja walk-forward)
===================================================================

Elo to rating predykcyjny aktualizowany wynikami meczow (inaczej niz ranking ATP,
ktory jest suma punktow turniejowych do rozstawiania). Liczymy go SAMI z danych
Sackmanna -- z natury sekwencyjny (expanding window), wiec leakage-safe.

Dwa ratingi per gracz:
  - elo        : ogolny (wszystkie nawierzchnie)
  - elo_surface: osobny per nawierzchnia (Clay/Grass/Hard)

K-factor dynamiczny (FiveThirtyEight): K = 250 / (n_played + 5)^0.4 -- nowicjusze
zmieniaja rating szybciej, weterani wolniej.

Cechy dodawane do modelu (antysymetryczne, OK dla symetryzacji):
  elo_diff, surface_elo_diff, elo_win_prob, surface_elo_win_prob

LEKCJA Z POPRZEDNIEGO CYKLU: nie ufamy pojedynczemu test setowi. Walidujemy
walk-forward przez 4 sezony (2021-2024) + McNemar parowany.
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

BASE_SCRIPT = Path(__file__).with_name("main_48_cech.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
TOUR = os.environ.get("TENNIS_TOUR", "atp")
HISTORY_START_YEAR = int(os.environ.get("TENNIS_HISTORY_START", "2001"))


def data_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


_wf_env = os.environ.get("TENNIS_WF_YEARS")
TARGET_YEARS = ([int(y) for y in _wf_env.split(",")] if _wf_env
                else [2020, 2021, 2022, 2023, 2024, 2025])
INIT_ELO = 1500.0
ELO_FEATURES = ["elo_diff", "surface_elo_diff", "elo_win_prob", "surface_elo_win_prob"]

ELO_COLS = ["w_elo", "l_elo", "w_surface_elo", "l_surface_elo"]


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


def k_factor(n_played: int) -> float:
    return 250.0 / ((n_played + 5) ** 0.4)


def build_elo_for_target_full(year: int) -> pd.DataFrame:
    """Przetwarza chronologicznie wszystkie mecze 2018..year, zwraca PELNA
    (przed dropna) ramke roku docelowego z kolumnami w_elo/l_elo/w_surface_elo/
    l_surface_elo = pre-match Elo (przed aktualizacja tym meczem)."""
    elo: dict[str, float] = {}
    elo_surf: dict[tuple, float] = {}
    n: dict[str, int] = {}
    n_surf: dict[tuple, int] = {}

    def ge(name):
        return elo.get(name, INIT_ELO)

    def ges(name, s):
        return elo_surf.get((name, s), INIT_ELO)

    def update(w, l, s, record):
        ew, el = ge(w), ge(l)
        esw, esl = ges(w, s), ges(l, s)
        if record is not None:
            record["w_elo"].append(ew)
            record["l_elo"].append(el)
            record["w_surface_elo"].append(esw)
            record["l_surface_elo"].append(esl)
        # ogolny
        exp_w = 1.0 / (1.0 + 10 ** ((el - ew) / 400.0))
        kw, kl = k_factor(n.get(w, 0)), k_factor(n.get(l, 0))
        elo[w] = ew + kw * (1.0 - exp_w)
        elo[l] = el + kl * (0.0 - (1.0 - exp_w))
        n[w] = n.get(w, 0) + 1
        n[l] = n.get(l, 0) + 1
        # per surface
        exp_sw = 1.0 / (1.0 + 10 ** ((esl - esw) / 400.0))
        ksw, ksl = k_factor(n_surf.get((w, s), 0)), k_factor(n_surf.get((l, s), 0))
        elo_surf[(w, s)] = esw + ksw * (1.0 - exp_sw)
        elo_surf[(l, s)] = esl + ksl * (0.0 - (1.0 - exp_sw))
        n_surf[(w, s)] = n_surf.get((w, s), 0) + 1
        n_surf[(l, s)] = n_surf.get((l, s), 0) + 1

    use = ["winner_name", "loser_name", "surface", "tourney_date", "match_num"]
    # historia: tylko stan Elo (record=None)
    for y in range(HISTORY_START_YEAR, year):
        df = pd.read_csv(data_file(y), usecols=use)
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
        df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
        for r in df.itertuples(index=False):
            update(r.winner_name, r.loser_name, r.surface, None)

    # rok docelowy: zapisujemy pre-match Elo
    target_full = pd.read_csv(data_file(year))
    target_full["tourney_date"] = pd.to_datetime(target_full["tourney_date"], format="%Y%m%d")
    target_full = target_full.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    rec = {c: [] for c in ELO_COLS}
    for r in target_full.itertuples(index=False):
        update(r.winner_name, r.loser_name, r.surface, rec)
    for c in ELO_COLS:
        target_full[c] = rec[c]
    return target_full


def add_elo_split(df_raw, target_elo_aligned, context_cols):
    df_raw = df_raw.copy().reset_index(drop=True)
    for col in ELO_COLS:
        df_raw[col] = target_elo_aligned[col].to_numpy()
    return df_raw


def elo_eval_for_year(ns: dict, year: int):
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

    # Elo dla pelnego roku, potem dropna(subset=cols_base) -> wyrownane do df_base.
    target_full = build_elo_for_target_full(year)
    target_aligned = target_full.dropna(subset=cols_base).reset_index(drop=True)
    assert len(target_aligned) == n_train + n_val + n_test, (
        f"Elo alignment {year}: {len(target_aligned)} vs {n_train + n_val + n_test}"
    )

    e_train = target_aligned.iloc[:n_train].reset_index(drop=True)
    e_val = target_aligned.iloc[n_train:n_train + n_val].reset_index(drop=True)
    e_test = target_aligned.iloc[n_train + n_val:].reset_index(drop=True)

    df_train_raw = add_elo_split(df_train_raw, e_train, ELO_COLS)
    df_val_raw = add_elo_split(df_val_raw, e_val, ELO_COLS)
    df_test_raw = add_elo_split(df_test_raw, e_test, ELO_COLS)
    raw_cols = ["match_id"] + ELO_COLS

    def build_split(df_raw, shuffle):
        sym = symmetrize_data(df_raw, shuffle=shuffle)
        sym = sym.merge(df_raw[raw_cols], on="match_id", how="left", validate="many_to_one")
        is_p1_winner = (sym["y"] == 1).to_numpy()
        p1_elo = np.where(is_p1_winner, sym["w_elo"], sym["l_elo"])
        p2_elo = np.where(is_p1_winner, sym["l_elo"], sym["w_elo"])
        p1_selo = np.where(is_p1_winner, sym["w_surface_elo"], sym["l_surface_elo"])
        p2_selo = np.where(is_p1_winner, sym["l_surface_elo"], sym["w_surface_elo"])
        sym["elo_diff"] = p1_elo - p2_elo
        sym["surface_elo_diff"] = p1_selo - p2_selo
        sym["elo_win_prob"] = 1.0 / (1.0 + 10 ** ((p2_elo - p1_elo) / 400.0))
        sym["surface_elo_win_prob"] = 1.0 / (1.0 + 10 ** ((p2_selo - p1_selo) / 400.0))
        return sym

    train_data = build_split(df_train_raw, shuffle=True)
    test_data = build_split(df_test_raw, shuffle=True)
    features = base_features + ELO_FEATURES

    elo_rf = RandomForestClassifier(
        **baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE
    )
    elo_rf.fit(train_data[features], train_data["y"])
    test_data["p1_win_probability"] = elo_rf.predict_proba(test_data[features])[:, 1]
    elo_eval, elo_match = compute_symmetric_match_evaluation(test_data)

    base_eval = ns["winner_perspective"][["match_id", "correct_prediction"]].copy()
    base_match = float(ns["match_accuracy"])

    importance = pd.DataFrame(
        {"feature": features, "importance": elo_rf.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1
    elo_ranks = {f: int(importance[importance["feature"] == f]["rank"].iloc[0]) for f in ELO_FEATURES}

    return base_eval, elo_eval[["match_id", "correct_prediction"]], base_match, elo_match, elo_ranks, len(features)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    z = (abs(b - c) - 1) / math.sqrt(n)
    return z, math.erfc(abs(z) / math.sqrt(2))


def main():
    rows = []
    all_pairs = []
    rank_acc = {f: [] for f in ELO_FEATURES}
    for year in TARGET_YEARS:
        print(f"\n===== ROK {year} =====", flush=True)
        ns = run_baseline_for_year(year)
        base_eval, elo_eval, base_match, elo_match, elo_ranks, nfeat = elo_eval_for_year(ns, year)
        merged = base_eval.merge(elo_eval, on="match_id", suffixes=("_base", "_elo"))
        for _, r in merged.iterrows():
            all_pairs.append((bool(r["correct_prediction_base"]), bool(r["correct_prediction_elo"])))
        for f, rk in elo_ranks.items():
            rank_acc[f].append(rk)
        rows.append({"year": year, "n": len(merged), "baseline": base_match,
                     "elo": elo_match, "delta": elo_match - base_match})
        print(f"  baseline={base_match:.4f}  +elo={elo_match:.4f}  delta={elo_match-base_match:+.4f}  "
              f"(elo feat ranks /{nfeat}: {elo_ranks})", flush=True)

    os.environ.pop("TENNIS_TARGET_YEAR", None)
    df = pd.DataFrame(rows)
    pairs = np.array(all_pairs)
    base_c, elo_c = pairs[:, 0], pairs[:, 1]
    N = len(pairs)
    b = int(np.sum(base_c & ~elo_c))
    c = int(np.sum(~base_c & elo_c))
    z, p = mcnemar(b, c)

    print("\n" + "=" * 72)
    print("WALK-FORWARD: baseline vs baseline + surface-adjusted ELO")
    print("=" * 72)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 72)
    print(f"POOLED ({N}): baseline={base_c.mean():.4f}  +elo={elo_c.mean():.4f}  "
          f"delta={elo_c.mean()-base_c.mean():+.4f}")
    print(f"  delta dodatnia w {int((df['delta']>0).sum())}/{len(df)} latach")
    print(f"  McNemar: b={b} c={c} z={z:.2f} p={p:.4f}")
    if p < 0.05 and c > b:
        print("  => ISTOTNE (p<0.05) na korzysc Elo.")
    else:
        print("  => Brak istotnosci (p>=0.05).")
    print()
    print("Sredni rank cech Elo (im nizszy tym wazniejsze):")
    for f in ELO_FEATURES:
        print(f"  {f:<22} sredni rank {np.mean(rank_acc[f]):.1f}")


if __name__ == "__main__":
    main()
