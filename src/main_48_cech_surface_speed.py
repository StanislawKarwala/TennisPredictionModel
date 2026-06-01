"""
Eksperyment Sprint 3a: Surface Speed Index + interakcja serve x speed
=====================================================================

Pomysl uzytkownika: szybszy kort -> gracze z mocniejszym serwem bardziej
faworyzowani. Implementacja:

1) court_pace_index -- proxy predkosci kortu liczony WYLACZNIE z historii
   2018-2023 (rozlacznie z ocenianym 2024 -> brak leakage). Dla kazdego turnieju
   (tourney_base = tourney_id bez prefiksu roku, + surface) bierzemy historyczny
   ace rate, centrujemy i skalujemy globalnie. Fallback dla turniejow o malym
   wsparciu (<20 meczow) lub nieobecnych w historii: srednia danej nawierzchni.

2) is_indoor -- flaga kortu zadaszonego (korty indoor sa szybsze). Kolumna 'indoor'
   istnieje w danych (O/I/NaN), ale baseline jej NIE uzywa.

3) interakcje serve x speed (kluczowe) -- przewaga serwisowa gracza pomnozona
   przez predkosc kortu. Antysymetryczne (znak zmienia sie przy p1<->p2), bo
   court_pace jest symetrycznym kontekstem meczu:
       ace_speed_diff   = (p1_ace_rate - p2_ace_rate) * court_pace_index
       first_won_speed_diff = (p1_first_won_pct - p2_first_won_pct) * court_pace_index

Porownanie: baseline RF (te same tuned HP) vs baseline + cechy surface speed.
Ablation: jedyna zmiana to nowe cechy. Metryka symetryczna z Sprint 1.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


BASE_SCRIPT = Path(__file__).with_name("main_48_cech.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
HISTORY_FILES = [DATA_DIR / f"{year}.csv" for year in (2018, 2019, 2020, 2021, 2022, 2023)]
YEAR_PREFIX = re.compile(r"^\d{4}-")
MIN_TOURNEY_SUPPORT = 20  # minimalna liczba meczow turnieju w historii dla wlasnego indeksu

NEW_FEATURES = [
    "court_pace_index",
    "is_indoor",
    "ace_speed_diff",
    "first_won_speed_diff",
]


def execute_base_pipeline_quietly() -> dict:
    original_cwd = os.getcwd()
    captured = io.StringIO()
    os.chdir(BASE_SCRIPT.parent)
    try:
        with contextlib.redirect_stdout(captured):
            return runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)


def tourney_base(tourney_id: object) -> str:
    return YEAR_PREFIX.sub("", str(tourney_id))


def build_court_pace_lookup() -> tuple[dict, dict, float, float]:
    """Buduje indeks predkosci kortu z historii 2018-2023 (BEZ 2024).

    Zwraca:
      pace_by_tourney: dict[(tourney_base, surface)] -> surowy ace rate
      pace_by_surface: dict[surface] -> sredni ace rate (fallback)
      global_mean, global_std: do centrowania/skalowania (z historii)
    """
    parts = []
    for path in HISTORY_FILES:
        df = pd.read_csv(path, usecols=["tourney_id", "surface", "w_ace", "l_ace", "w_svpt", "l_svpt"])
        parts.append(df)
    hist = pd.concat(parts, ignore_index=True)
    hist["tot_ace"] = hist["w_ace"] + hist["l_ace"]
    hist["tot_svpt"] = hist["w_svpt"] + hist["l_svpt"]
    hist = hist[hist["tot_svpt"] > 0].copy()
    hist["tourney_base"] = hist["tourney_id"].map(tourney_base)

    # ace rate per turniej (z minimalnym wsparciem)
    grp = hist.groupby(["tourney_base", "surface"]).agg(
        ace=("tot_ace", "sum"), svpt=("tot_svpt", "sum"), n=("tot_ace", "size")
    )
    grp["ace_rate"] = grp["ace"] / grp["svpt"]
    supported = grp[grp["n"] >= MIN_TOURNEY_SUPPORT]
    pace_by_tourney = {idx: float(row["ace_rate"]) for idx, row in supported.iterrows()}

    # fallback per surface
    surf = hist.groupby("surface").agg(ace=("tot_ace", "sum"), svpt=("tot_svpt", "sum"))
    pace_by_surface = {s: float(r["ace"] / r["svpt"]) for s, r in surf.iterrows()}

    # globalne statystyki do centrowania (z surowych ace rate per mecz, wazone svpt)
    global_mean = float(hist["tot_ace"].sum() / hist["tot_svpt"].sum())
    # std liczymy na poziomie turniejow (rozrzut predkosci kortow)
    global_std = float(supported["ace_rate"].std()) or 0.02

    return pace_by_tourney, pace_by_surface, global_mean, global_std


def court_pace_index(tourney_id, surface, lookup) -> float:
    pace_by_tourney, pace_by_surface, gmean, gstd = lookup
    key = (tourney_base(tourney_id), surface)
    raw = pace_by_tourney.get(key)
    if raw is None:
        raw = pace_by_surface.get(surface, gmean)
    return (raw - gmean) / gstd


def main() -> None:
    ns = execute_base_pipeline_quietly()
    symmetrize_data = ns["symmetrize_data"]
    compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
    evaluate_calibration_quality = ns["evaluate_calibration_quality"]
    baseline_search = ns["search"]
    RANDOM_STATE = ns["RANDOM_STATE"]
    base_features = list(ns["features"])

    df_train_raw = ns["df_train_raw"].copy()
    df_val_raw = ns["df_val_raw"].copy()
    df_test_raw = ns["df_test_raw"].copy()

    baseline_val_acc = float(ns["val_acc"])
    baseline_test_acc = float(ns["test_acc"])
    baseline_match_acc = float(ns["match_accuracy"])

    print("Buduje court_pace_index z historii 2018-2023...")
    lookup = build_court_pace_lookup()

    # df_*_raw maja kolumny tourney_date, surface (z cols_base po Sprint 1).
    # Potrzebujemy tourney_id i indoor -- doczytujemy z 2024.csv i laczymy
    # POZYCYJNIE (df_*_raw to kolejne kawalki 2024 w tej samej kolejnosci po
    # tym samym dropna i sortowaniu co baseline).
    full_2024 = pd.read_csv(DATA_DIR / "2024.csv")
    full_2024["tourney_date"] = pd.to_datetime(full_2024["tourney_date"], format="%Y%m%d")
    full_2024 = full_2024.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    cols_base = list(ns["cols_base"])
    full_2024_base = full_2024[cols_base + ["tourney_id", "indoor"]].dropna(subset=cols_base).reset_index(drop=True)

    n_train, n_val, n_test = len(df_train_raw), len(df_val_raw), len(df_test_raw)
    assert len(full_2024_base) == n_train + n_val + n_test, (
        f"Niespojnosc dlugosci 2024: {len(full_2024_base)} vs "
        f"{n_train + n_val + n_test}"
    )
    ctx_train = full_2024_base.iloc[:n_train].reset_index(drop=True)
    ctx_val = full_2024_base.iloc[n_train:n_train + n_val].reset_index(drop=True)
    ctx_test = full_2024_base.iloc[n_train + n_val:].reset_index(drop=True)

    def attach_context(df_raw, ctx):
        df_raw = df_raw.copy()
        df_raw["match_id"] = range(len(df_raw))
        ctx = ctx.copy()
        ctx["match_id"] = range(len(ctx))
        cpi = [court_pace_index(t, s, lookup) for t, s in zip(ctx["tourney_id"], ctx["surface"])]
        is_indoor = (ctx["indoor"].astype(str) == "I").astype(int).to_numpy()
        ctx_small = pd.DataFrame({
            "match_id": ctx["match_id"],
            "court_pace_index": cpi,
            "is_indoor": is_indoor,
        })
        return df_raw.merge(ctx_small, on="match_id", how="left", validate="one_to_one")

    df_train_raw = attach_context(df_train_raw, ctx_train)
    df_val_raw = attach_context(df_val_raw, ctx_val)
    df_test_raw = attach_context(df_test_raw, ctx_test)

    def build_split(df_raw, shuffle):
        sym = symmetrize_data(df_raw, shuffle=shuffle)
        # Dolacz court_pace_index + is_indoor po match_id (kontekst symetryczny).
        ctx_small = df_raw[["match_id", "court_pace_index", "is_indoor"]]
        sym = sym.merge(ctx_small, on="match_id", how="left", validate="many_to_one")
        # Interakcje serve x speed (z juz zsymetryzowanych p1_/p2_ cech serwisowych).
        sym["ace_speed_diff"] = (sym["p1_ace_rate"] - sym["p2_ace_rate"]) * sym["court_pace_index"]
        sym["first_won_speed_diff"] = (
            (sym["p1_first_won_pct"] - sym["p2_first_won_pct"]) * sym["court_pace_index"]
        )
        return sym

    train_data = build_split(df_train_raw, shuffle=True)
    val_data = build_split(df_val_raw, shuffle=True)
    test_data = build_split(df_test_raw, shuffle=True)

    features = base_features + NEW_FEATURES

    X_train, y_train = train_data[features], train_data["y"]
    X_val, y_val = val_data[features], val_data["y"]
    X_test, y_test = test_data[features], test_data["y"]

    best_rf = RandomForestClassifier(
        **baseline_search.best_params_,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    print(f"Liczba cech: {len(features)} (baseline: {len(base_features)}, nowe: {len(NEW_FEATURES)})")
    print("Trening RF z tuned HP baseline...")
    best_rf.fit(X_train, y_train)

    val_acc = float(accuracy_score(y_val, best_rf.predict(X_val)))
    test_acc = float(accuracy_score(y_test, best_rf.predict(X_test)))
    proba_test = best_rf.predict_proba(X_test)[:, 1]
    test_data["p1_win_probability"] = proba_test
    _, match_acc = compute_symmetric_match_evaluation(test_data)
    quality = evaluate_calibration_quality(y_test.to_numpy(), proba_test)

    # waznosc nowych cech
    importance = pd.DataFrame(
        {"feature": features, "importance": best_rf.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1

    print("\n" + "=" * 78)
    print("POROWNANIE: baseline RF vs baseline + SURFACE SPEED")
    print("=" * 78)
    print(f"{'':<16} val      test     match    Brier")
    print(f"{'baseline':<16} {baseline_val_acc:.4f}   {baseline_test_acc:.4f}   {baseline_match_acc:.4f}")
    print(f"{'+ surface_speed':<16} {val_acc:.4f}   {test_acc:.4f}   {match_acc:.4f}   {quality['brier_score']:.4f}")
    print("-" * 78)
    print(
        f"DELTA: val={val_acc - baseline_val_acc:+.4f}  "
        f"test={test_acc - baseline_test_acc:+.4f}  "
        f"match={match_acc - baseline_match_acc:+.4f}"
    )
    print()
    print("Pozycja nowych cech w waznosci (feature importance):")
    for feat in NEW_FEATURES:
        row = importance[importance["feature"] == feat].iloc[0]
        print(f"  {feat:<22} rank {int(row['rank']):>2}/{len(features)}  importance={row['importance']:.4f}")
    print()
    print(
        "UWAGA: CI dla match accuracy ~ +/-4 p.p. przy 590 meczach. Ostateczna "
        "ocena w Sprint 4 (walk-forward)."
    )


if __name__ == "__main__":
    main()
