"""
Eksperyment Sprint 3c: EWMA (recency weighting) zamiast SMA
===========================================================

Baseline liczy forme i statystyki serwisowe jako prosta srednia z ostatnich 10
meczow (SMA) w oknie 365 dni (po A3). EWMA to gladsza alternatywa: kazdy starszy
mecz ma wykladniczo mniejsza wage, bez twardego okna:

    EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}

alpha = 0.18 odpowiada mniej wiecej "ostatnim 10 meczom" (2/(10+1)), ale z
plynnym zanikiem. Cold-start: EWMA inicjalizowane priorem (forma 0.5,
SERVE_DEFAULTS), aktualizowane incrementalnie w O(N) przez cala chronologie
(historia 2018-2023 najpierw, potem 2024 -- bez leakage).

Ablation: nadpisujemy TYLKO kolumny formy/serwisu w df_*_raw (reszta bez zmian),
re-symetryzujemy, trenujemy RF z tymi samymi tuned HP. Metryka symetryczna (Sprint 1).
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


BASE_SCRIPT = Path(__file__).with_name("main_48_cech.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
TOUR = os.environ.get("TENNIS_TOUR", "atp")
TARGET_YEAR = int(os.environ.get("TENNIS_TARGET_YEAR", "2025"))
HISTORY_START_YEAR = int(os.environ.get("TENNIS_HISTORY_START", "2001"))


def data_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


HISTORY_FILES = [data_file(y) for y in range(HISTORY_START_YEAR, TARGET_YEAR)]

ALPHA = 0.18  # ~ "ostatnie 10 meczow", ale z plynnym zanikiem

SERVE_STAT_NAMES = ["ace_rate", "df_rate", "first_in_pct", "first_won_pct",
                    "second_won_pct", "bp_save_pct", "bp_faced_per_game", "return_pts_won"]
SERVE_DEFAULTS = {
    "ace_rate": 0.08, "df_rate": 0.03, "first_in_pct": 0.60, "first_won_pct": 0.70,
    "second_won_pct": 0.50, "bp_save_pct": 0.60, "bp_faced_per_game": 0.40, "return_pts_won": 0.35,
}

# Kolumny nadpisywane w df_*_raw (te, ktore baseline liczy jako SMA).
OVERWRITE_COLS = (
    ["w_form", "l_form", "w_surface_form", "l_surface_form"]
    + [f"w_{s}" for s in SERVE_STAT_NAMES]
    + [f"l_{s}" for s in SERVE_STAT_NAMES]
)


def execute_base_pipeline_quietly() -> dict:
    original_cwd = os.getcwd()
    captured = io.StringIO()
    os.chdir(BASE_SCRIPT.parent)
    try:
        with contextlib.redirect_stdout(captured):
            return runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)


def per_match_serve_stats(row, is_winner: bool) -> dict:
    """Statystyki serwisowe z POJEDYNCZEGO meczu z perspektywy gracza.
    Zwraca tylko te, ktorych mianownik > 0 (reszta -> None = pomijamy update)."""
    p = "w" if is_winner else "l"
    o = "l" if is_winner else "w"
    g = lambda c: float(row[c])
    svpt = g(f"{p}_svpt")
    first_in = g(f"{p}_1stIn")
    sv_gms = g(f"{p}_SvGms")
    bp_faced = g(f"{p}_bpFaced")
    opp_svpt = g(f"{o}_svpt")
    out = {}
    out["ace_rate"] = g(f"{p}_ace") / svpt if svpt > 0 else None
    out["df_rate"] = g(f"{p}_df") / svpt if svpt > 0 else None
    out["first_in_pct"] = first_in / svpt if svpt > 0 else None
    out["first_won_pct"] = g(f"{p}_1stWon") / first_in if first_in > 0 else None
    second_serve = svpt - first_in
    out["second_won_pct"] = g(f"{p}_2ndWon") / second_serve if second_serve > 0 else None
    out["bp_save_pct"] = g(f"{p}_bpSaved") / bp_faced if bp_faced > 0 else None
    out["bp_faced_per_game"] = bp_faced / sv_gms if sv_gms > 0 else None
    if opp_svpt > 0:
        out["return_pts_won"] = (opp_svpt - g(f"{o}_1stWon") - g(f"{o}_2ndWon")) / opp_svpt
    else:
        out["return_pts_won"] = None
    return out


def compute_ewma_features(full_2024_base: pd.DataFrame, cols_base: list[str]) -> pd.DataFrame:
    """Liczy EWMA formy/serwisu/surface_form dla meczow 2024, przetwarzajac
    chronologicznie historie 2018-2023 + 2024. Zwraca ramke 1:1 do full_2024_base
    z kolumnami w_*/l_* (te same nazwy co baseline)."""
    serve_cols = [c for c in cols_base if c.startswith(("w_", "l_"))]
    keep = ["winner_name", "loser_name", "surface"] + serve_cols

    parts = []
    for path in HISTORY_FILES:
        df = pd.read_csv(path)
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
        df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
        parts.append(df[keep].dropna())
    history = pd.concat(parts, ignore_index=True)
    full_seq = pd.concat([history, full_2024_base[keep]], ignore_index=True)
    start_idx = len(history)

    # Stany EWMA per gracz.
    form_state: dict[str, float] = {}
    serve_state: dict[str, dict] = {}
    surf_state: dict[tuple, float] = {}

    def get_form(name):
        return form_state.get(name, 0.5)

    def get_serve(name):
        return serve_state.get(name, SERVE_DEFAULTS).copy()

    def get_surf(name, surface):
        return surf_state.get((name, surface), get_form(name))

    rows = full_seq.to_dict("records")
    out = {c: [] for c in OVERWRITE_COLS}

    for i, row in enumerate(rows):
        w, l, surf = row["winner_name"], row["loser_name"], row["surface"]

        if i >= start_idx:  # mecz 2024 -- zapisz pre-match EWMA
            out["w_form"].append(get_form(w))
            out["l_form"].append(get_form(l))
            out["w_surface_form"].append(get_surf(w, surf))
            out["l_surface_form"].append(get_surf(l, surf))
            sw, sl = get_serve(w), get_serve(l)
            for s in SERVE_STAT_NAMES:
                out[f"w_{s}"].append(sw[s])
                out[f"l_{s}"].append(sl[s])

        # update form
        form_state[w] = ALPHA * 1.0 + (1 - ALPHA) * get_form(w)
        form_state[l] = ALPHA * 0.0 + (1 - ALPHA) * get_form(l)
        # update surface form
        surf_state[(w, surf)] = ALPHA * 1.0 + (1 - ALPHA) * get_surf(w, surf)
        surf_state[(l, surf)] = ALPHA * 0.0 + (1 - ALPHA) * get_surf(l, surf)
        # update serve (tylko statystyki o sensownym mianowniku)
        for name, is_win in ((w, True), (l, False)):
            cur = get_serve(name)
            stats = per_match_serve_stats(row, is_win)
            for s in SERVE_STAT_NAMES:
                x = stats[s]
                if x is not None and np.isfinite(x):
                    cur[s] = ALPHA * x + (1 - ALPHA) * cur[s]
            serve_state[name] = cur

    return pd.DataFrame(out)


def main() -> None:
    ns = execute_base_pipeline_quietly()
    symmetrize_data = ns["symmetrize_data"]
    compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
    evaluate_calibration_quality = ns["evaluate_calibration_quality"]
    baseline_search = ns["search"]
    RANDOM_STATE = ns["RANDOM_STATE"]
    features = list(ns["features"])
    cols_base = list(ns["cols_base"])

    df_train_raw = ns["df_train_raw"].copy()
    df_val_raw = ns["df_val_raw"].copy()
    df_test_raw = ns["df_test_raw"].copy()
    baseline_val_acc = float(ns["val_acc"])
    baseline_test_acc = float(ns["test_acc"])
    baseline_match_acc = float(ns["match_accuracy"])

    full_target = pd.read_csv(data_file(TARGET_YEAR))
    full_target["tourney_date"] = pd.to_datetime(full_target["tourney_date"], format="%Y%m%d")
    full_target = full_target.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    full_target_base = full_target[cols_base].dropna(subset=cols_base).reset_index(drop=True)

    n_train, n_val, n_test = len(df_train_raw), len(df_val_raw), len(df_test_raw)
    assert len(full_target_base) == n_train + n_val + n_test

    print(f"Licze cechy EWMA (forma/serwis/surface_form) z chronologii {HISTORY_START_YEAR}-{TARGET_YEAR}...")
    ewma = compute_ewma_features(full_target_base, cols_base)
    e_train = ewma.iloc[:n_train].reset_index(drop=True)
    e_val = ewma.iloc[n_train:n_train + n_val].reset_index(drop=True)
    e_test = ewma.iloc[n_train + n_val:].reset_index(drop=True)

    def overwrite(df_raw, e):
        df_raw = df_raw.copy().reset_index(drop=True)
        for col in OVERWRITE_COLS:
            df_raw[col] = e[col].to_numpy()
        return df_raw

    df_train_raw = overwrite(df_train_raw, e_train)
    df_val_raw = overwrite(df_val_raw, e_val)
    df_test_raw = overwrite(df_test_raw, e_test)

    train_data = symmetrize_data(df_train_raw, shuffle=True)
    val_data = symmetrize_data(df_val_raw, shuffle=True)
    test_data = symmetrize_data(df_test_raw, shuffle=True)

    X_train, y_train = train_data[features], train_data["y"]
    X_val, y_val = val_data[features], val_data["y"]
    X_test, y_test = test_data[features], test_data["y"]

    best_rf = RandomForestClassifier(
        **baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE
    )
    print(f"Liczba cech: {len(features)} (te same nazwy co baseline; forma/serwis -> EWMA)")
    best_rf.fit(X_train, y_train)

    val_acc = float(accuracy_score(y_val, best_rf.predict(X_val)))
    test_acc = float(accuracy_score(y_test, best_rf.predict(X_test)))
    proba_test = best_rf.predict_proba(X_test)[:, 1]
    test_data["p1_win_probability"] = proba_test
    _, match_acc = compute_symmetric_match_evaluation(test_data)
    quality = evaluate_calibration_quality(y_test.to_numpy(), proba_test)

    print("\n" + "=" * 78)
    print("POROWNANIE: baseline RF (SMA + okno 365d) vs EWMA (recency weighting)")
    print("=" * 78)
    print(f"{'':<16} val      test     match    Brier")
    print(f"{'baseline (SMA)':<16} {baseline_val_acc:.4f}   {baseline_test_acc:.4f}   {baseline_match_acc:.4f}")
    print(f"{'EWMA':<16} {val_acc:.4f}   {test_acc:.4f}   {match_acc:.4f}   {quality['brier_score']:.4f}")
    print("-" * 78)
    print(
        f"DELTA: val={val_acc - baseline_val_acc:+.4f}  "
        f"test={test_acc - baseline_test_acc:+.4f}  "
        f"match={match_acc - baseline_match_acc:+.4f}"
    )
    print()
    print(f"alpha={ALPHA} (half-life ~ {np.log(0.5)/np.log(1-ALPHA):.1f} meczow)")
    print("UWAGA: CI dla match accuracy ~ +/-4 p.p. przy 590 meczach.")


if __name__ == "__main__":
    main()
