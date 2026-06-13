"""
Eksperyment: kursy bukmacherskie jako cechy modelu (walk-forward)
=================================================================

Pytanie: czy kursy zamkniecia (zagregowana wiedza rynku) przebijaja sufit ~65%,
ktorego nie ruszyly Elo / fatigue / surface speed / warianty slice-aware?

Kursy pochodza z data/odds/atp_odds_{rok}.csv (pipeline: src/fetch_odds.py --
zagraniczne EnglishAvg/Pinnacle/B365/Max z tennis-data; src/fetch_odds_betexplorer.py
-- polscy bukmacherzy z BetExplorer). Kurs zamkniecia to informacja dostepna
PRZED meczem -- brak leakage. Kaskada wyboru pary kursow per mecz:
EnglishAvg -> Pinnacle -> B365 -> PolishAvg (srednia polskich bukmacherow).

Cechy (4, dolaczane do 40 cech baseline):
  p1_implied_prob, p2_implied_prob -- prawdopodobienstwa implikowane kursami,
      znormalizowane do sumy 1 (usuniecie marzy bukmacherskiej / vig):
      p1 = (1/kurs_p1) / (1/kurs_p1 + 1/kurs_p2)
  implied_prob_diff               -- p1 - p2 (antysymetryczna)
  has_odds                        -- 1 gdy mecz ma kursy; mecze bez kursow
      (~5-10%: czesc Davis Cup/igrzysk itp.) dostaja neutralne 0.5/0.5/0.0,
      dzieki czemu zbior testowy jest IDENTYCZNY z baseline (test parowany).

Benchmarki (na tych samych meczach):
  baseline (40 cech)  vs  baseline + kursy (44)  vs  sam rynek (faworyt wg
  kursow; accuracy liczona na podzbiorze meczow z kursami + Brier rynku).

Walidacja jak w tennis_model_elo.py: walk-forward po sezonach (default
2020-2025, env TENNIS_WF_YEARS), te same tuned HP co baseline (delta wynika
z cech, nie ze strojenia), metryka symetryczna match-level, pooled McNemar.
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

BASE_SCRIPT = Path(__file__).with_name("tennis_model.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
ODDS_DIR = BASE_DIR / "data" / "odds"
TOUR = os.environ.get("TENNIS_TOUR", "atp")

# Kursy dostepne od sezonu 2017 (tennis-data + BetExplorer), wiec walk-forward
# obejmuje tu 9 sezonow -- wiecej niz w pozostalych eksperymentach (2020-2025).
_wf_env = os.environ.get("TENNIS_WF_YEARS")
TARGET_YEARS = ([int(y) for y in _wf_env.split(",")] if _wf_env
                else [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])

ODDS_FEATURES = ["p1_implied_prob", "p2_implied_prob", "implied_prob_diff", "has_odds"]

# Kaskada zrodel kursow (kolumny w atp_odds_{rok}.csv): od najszerszego pokrycia.
ODDS_PRIORITY = [("EnglishAvg_winner", "EnglishAvg_loser"),
                 ("Avg_winner", "Avg_loser"),  # stara nazwa (pliki sprzed migracji)
                 ("PS_winner", "PS_loser"),
                 ("B365_winner", "B365_loser")]
POLISH_PREFIXES = ["STS", "Fortuna", "Superbet", "Betclic", "BETFAN", "LVBET", "Fuksiarz"]

ODDS_RAW_COLS = ["odds_w", "odds_l"]


def data_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


def odds_file(year: int) -> Path:
    return ODDS_DIR / f"{TOUR}_odds_{year}.csv"


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


def load_odds_pairs(year: int) -> pd.DataFrame:
    """match_key -> (odds_w, odds_l) wg kaskady zrodel; NaN gdy brak kursow."""
    fp = odds_file(year)
    if not fp.exists():
        raise FileNotFoundError(
            f"Brak {fp} -- najpierw uruchom: python src/fetch_odds.py --years {year}"
        )
    odds = pd.read_csv(fp)
    odds_w = pd.Series(np.nan, index=odds.index)
    odds_l = pd.Series(np.nan, index=odds.index)
    for w_col, l_col in ODDS_PRIORITY:
        if w_col in odds.columns and l_col in odds.columns:
            take = odds_w.isna() & odds[w_col].notna() & odds[l_col].notna()
            odds_w[take] = odds[w_col][take]
            odds_l[take] = odds[l_col][take]
    # Fallback: srednia z dostepnych polskich bukmacherow (np. United Cup,
    # ktorego nie ma w tennis-data, a jest na BetExplorer).
    polish_w = [f"{p}_winner" for p in POLISH_PREFIXES if f"{p}_winner" in odds.columns]
    polish_l = [f"{p}_loser" for p in POLISH_PREFIXES if f"{p}_loser" in odds.columns]
    if polish_w and polish_l:
        mean_w = odds[polish_w].mean(axis=1)
        mean_l = odds[polish_l].mean(axis=1)
        take = odds_w.isna() & mean_w.notna() & mean_l.notna()
        odds_w[take] = mean_w[take]
        odds_l[take] = mean_l[take]
    return pd.DataFrame({"match_key": odds["match_key"].astype(str),
                         "odds_w": odds_w, "odds_l": odds_l})


def build_odds_for_target_full(year: int) -> pd.DataFrame:
    """Pelna (przed dropna) ramka sezonu z kolumnami odds_w/odds_l --
    sortowanie identyczne z baseline -> wyrownanie pozycyjne po dropna."""
    target_full = pd.read_csv(data_file(year))
    target_full["tourney_date"] = pd.to_datetime(target_full["tourney_date"], format="%Y%m%d")
    target_full = target_full.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    target_full["match_key"] = (target_full["tourney_id"].astype(str) + "_"
                                + target_full["match_num"].astype(str))
    pairs = load_odds_pairs(year)
    target_full = target_full.merge(pairs, on="match_key", how="left", validate="one_to_one")
    return target_full


def add_odds_split(df_raw, target_aligned, context_cols=ODDS_RAW_COLS):
    df_raw = df_raw.copy().reset_index(drop=True)
    for col in context_cols:
        df_raw[col] = target_aligned[col].to_numpy()
    return df_raw


def attach_odds_features(sym: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Po symetryzacji: mapuje kursy zwyciezcy/przegranego na p1/p2 i liczy
    znormalizowane prawdopodobienstwa implikowane. Mecze bez kursow dostaja
    neutralne 0.5/0.5/0.0 + has_odds=0 (ten sam test set co baseline)."""
    sym = sym.merge(df_raw[["match_id"] + ODDS_RAW_COLS], on="match_id",
                    how="left", validate="many_to_one")
    is_p1_winner = (sym["y"] == 1).to_numpy()
    odds_w = sym["odds_w"].to_numpy(dtype=float)
    odds_l = sym["odds_l"].to_numpy(dtype=float)
    has = np.isfinite(odds_w) & np.isfinite(odds_l) & (odds_w > 1.0) & (odds_l > 1.0)

    q_w = np.where(has, 1.0 / np.where(has, odds_w, 1.0), np.nan)
    q_l = np.where(has, 1.0 / np.where(has, odds_l, 1.0), np.nan)
    p_winner = q_w / (q_w + q_l)  # znormalizowane P(zwyciezca) wg rynku

    p1 = np.where(is_p1_winner, p_winner, 1.0 - p_winner)
    p1 = np.where(has, p1, 0.5)
    sym["p1_implied_prob"] = p1
    sym["p2_implied_prob"] = 1.0 - p1
    sym["implied_prob_diff"] = sym["p1_implied_prob"] - sym["p2_implied_prob"]
    sym["has_odds"] = has.astype(int)
    return sym


def market_benchmark(df_test_raw: pd.DataFrame) -> dict:
    """Sam rynek na meczach testowych Z kursami: accuracy faworyta + Brier.

    df_test_raw ma odds_w/odds_l (kurs zwyciezcy/przegranego) -- rynek 'trafia',
    gdy kurs zwyciezcy < kurs przegranego (byl faworytem)."""
    odds_w = df_test_raw["odds_w"].to_numpy(dtype=float)
    odds_l = df_test_raw["odds_l"].to_numpy(dtype=float)
    has = np.isfinite(odds_w) & np.isfinite(odds_l) & (odds_w > 1.0) & (odds_l > 1.0)
    if not has.any():
        return {"coverage": 0.0, "accuracy": float("nan"), "brier": float("nan"), "n": 0}
    q_w, q_l = 1.0 / odds_w[has], 1.0 / odds_l[has]
    p_winner = q_w / (q_w + q_l)
    return {
        "coverage": float(has.mean()),
        "accuracy": float((p_winner > 0.5).mean()),
        "brier": float(np.mean((1.0 - p_winner) ** 2)),
        "n": int(has.sum()),
    }


def odds_eval_for_year(ns: dict, year: int):
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

    target_full = build_odds_for_target_full(year)
    target_aligned = target_full.dropna(subset=cols_base).reset_index(drop=True)
    assert len(target_aligned) == n_train + n_val + n_test, (
        f"Odds alignment {year}: {len(target_aligned)} vs {n_train + n_val + n_test}"
    )

    e_train = target_aligned.iloc[:n_train].reset_index(drop=True)
    e_val = target_aligned.iloc[n_train:n_train + n_val].reset_index(drop=True)
    e_test = target_aligned.iloc[n_train + n_val:].reset_index(drop=True)

    df_train_raw = add_odds_split(df_train_raw, e_train)
    df_val_raw = add_odds_split(df_val_raw, e_val)
    df_test_raw = add_odds_split(df_test_raw, e_test)

    def build_split(df_raw, shuffle):
        sym = symmetrize_data(df_raw, shuffle=shuffle)
        return attach_odds_features(sym, df_raw)

    train_data = build_split(df_train_raw, shuffle=True)
    test_data = build_split(df_test_raw, shuffle=True)
    features = base_features + ODDS_FEATURES

    odds_rf = RandomForestClassifier(
        **baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE
    )
    odds_rf.fit(train_data[features], train_data["y"])
    test_data["p1_win_probability"] = odds_rf.predict_proba(test_data[features])[:, 1]
    odds_eval, odds_match = compute_symmetric_match_evaluation(test_data)

    base_eval = ns["winner_perspective"][["match_id", "correct_prediction"]].copy()
    base_match = float(ns["match_accuracy"])

    market = market_benchmark(df_test_raw)

    importance = pd.DataFrame(
        {"feature": features, "importance": odds_rf.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1
    odds_ranks = {f: int(importance[importance["feature"] == f]["rank"].iloc[0])
                  for f in ODDS_FEATURES}

    return (base_eval, odds_eval[["match_id", "correct_prediction"]],
            base_match, odds_match, market, odds_ranks, len(features))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    # Korekta ciaglosci nie moze zejsc ponizej zera -- dla b==c byloby z<0,
    # a abs(z) dawalby p<1 zamiast poprawnego p=1 (brak roznicy).
    z = max(abs(b - c) - 1, 0) / math.sqrt(n)
    return z, math.erfc(z / math.sqrt(2))


def main():
    rows = []
    all_pairs = []
    rank_acc = {f: [] for f in ODDS_FEATURES}
    market_rows = []

    for year in TARGET_YEARS:
        print(f"\n===== ROK {year} =====", flush=True)
        ns = run_baseline_for_year(year)
        (base_eval, odds_eval, base_match, odds_match,
         market, odds_ranks, nfeat) = odds_eval_for_year(ns, year)

        merged = base_eval.merge(odds_eval, on="match_id", suffixes=("_base", "_odds"))
        for _, r in merged.iterrows():
            all_pairs.append((bool(r["correct_prediction_base"]),
                              bool(r["correct_prediction_odds"])))
        rows.append({"year": year, "n": len(merged), "baseline": base_match,
                     "odds": odds_match, "delta": odds_match - base_match})
        market_rows.append({"year": year, **market})
        for f, rk in odds_ranks.items():
            rank_acc[f].append(rk)
        print(f"  baseline={base_match:.4f}  +odds={odds_match:.4f}  "
              f"delta={odds_match - base_match:+.4f}", flush=True)
        print(f"  rynek (faworyt wg kursow): acc={market['accuracy']:.4f} "
              f"Brier={market['brier']:.4f} na {market['n']} meczach "
              f"({market['coverage'] * 100:.1f}% pokrycia)", flush=True)
        print(f"  rangi cech kursowych /{nfeat}: {odds_ranks}", flush=True)

    os.environ.pop("TENNIS_TARGET_YEAR", None)

    df = pd.DataFrame(rows)
    pairs = np.array(all_pairs)
    bc, oc = pairs[:, 0], pairs[:, 1]
    b = int(np.sum(bc & ~oc))
    c = int(np.sum(~bc & oc))
    z, p = mcnemar(b, c)
    pos = int((df["delta"] > 0).sum())

    print("\n" + "=" * 72)
    print("WALK-FORWARD: baseline vs baseline + kursy bukmacherskie")
    print("=" * 72)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 72)
    print(f"POOLED ({len(pairs)}): baseline={bc.mean():.4f}  +odds={oc.mean():.4f}  "
          f"delta={oc.mean() - bc.mean():+.4f}")
    print(f"  delta dodatnia w {pos}/{len(df)} sezonach")
    verdict = ("ISTOTNE (na korzysc kursow)" if (p < 0.05 and c > b)
               else "ISTOTNE (na niekorzysc)" if p < 0.05
               else "brak istotnosci (p>=0.05)")
    print(f"  McNemar: b={b} c={c} z={z:.2f} p={p:.4f}  -> {verdict}")

    mdf = pd.DataFrame(market_rows)
    print("\nSam rynek (faworyt wg kursow, na meczach z kursami):")
    print(mdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    total_n = int(mdf["n"].sum())
    if total_n:
        w_acc = float((mdf["accuracy"] * mdf["n"]).sum() / total_n)
        w_brier = float((mdf["brier"] * mdf["n"]).sum() / total_n)
        print(f"  POOLED rynek: acc={w_acc:.4f}  Brier={w_brier:.4f}  (n={total_n})")

    print("\nSredni rank cech kursowych (im nizszy tym wazniejsze):")
    for f in ODDS_FEATURES:
        print(f"  {f:22s} sredni rank {np.mean(rank_acc[f]):.1f}")


if __name__ == "__main__":
    main()
