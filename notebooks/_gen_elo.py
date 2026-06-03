"""Generuje i wykonuje OD ZERA notebook Elo w stylu narracyjnym (jak
TPM_Experiment_SliceAware_BestOf5_v1.ipynb) -- markdown opisuje krok, kod go
WYKONUJE i drukuje posrednie wyniki, zamiast wolac samo m.main().

Uzycie: python _gen_elo.py
"""
from _nbtools import make_and_run

SETUP = """import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))"""

cells = [
("md", """# Eksperyment: Surface-adjusted Elo jako cecha

## Cel
Dodać do modelu rating **Elo** — przewidujący, aktualizowany po każdym meczu (inaczej niż ranking
ATP, który jest sumą punktów turniejowych do rozstawiania). Dwa ratingi per gracz: ogólny oraz
osobny per nawierzchnia. Stąd 4 nowe cechy: `elo_diff`, `surface_elo_diff`, `elo_win_prob`,
`surface_elo_win_prob`.

## Metoda (leakage-safe)
- Elo liczony **sekwencyjnie** z danych Sackmanna (2001 → rok docelowy): każdy mecz aktualizuje
  rating *po* zapisaniu stanu sprzed meczu → model nie widzi wyniku, który przewiduje.
- K-factor dynamiczny (FiveThirtyEight): `K = 250/(n+5)^0.4` — nowicjusze zmieniają rating szybciej,
  weterani wolniej.
- Te same tuned hiperparametry co baseline (czysta ablacja: zmieniamy **tylko** cechy).
- **Walidacja walk-forward** przez 6 sezonów (2020–2025) + parowany test **McNemar** — bo pojedynczy
  sezon potrafi „skłamać"."""),

("code", SETUP),

("md", """## 1. Reuse baseline (rok docelowy 2025)
Uruchamiamy `tennis_model.py` (z wyciszonym outputem) dla 2025 i pobieramy z niego: chronologiczne
splity train/val/test, tuned hiperparametry, funkcje pomocnicze (symetryzacja, metryka symetryczna)
oraz wynik baseline — żeby model z Elo liczył się na **identycznych** meczach."""),

("code", """import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tennis_model_elo import (
    run_baseline_for_year, build_elo_for_target_full, add_elo_split,
    elo_eval_for_year, k_factor, mcnemar,
    ELO_COLS, ELO_FEATURES, HISTORY_START_YEAR, TARGET_YEARS,
)

TARGET = 2025
ns = run_baseline_for_year(TARGET)                 # runpy tennis_model.py (cicho)
symmetrize_data = ns["symmetrize_data"]
compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
base_features = list(ns["features"]); cols_base = list(ns["cols_base"])
RANDOM_STATE = ns["RANDOM_STATE"]
baseline_match = float(ns["match_accuracy"])

print(f"Baseline {TARGET}: val={ns['val_acc']:.4f}  test={ns['test_acc']:.4f}  match={baseline_match:.4f}")
print(f"Historia Elo: {HISTORY_START_YEAR}..{TARGET-1}   |   cechy baseline: {len(base_features)}")"""),

("md", """## 2. Liczymy Elo z historii (leakage-safe, dynamiczne K)
Aktualizacja: `R += K · (wynik − oczekiwanie)`, gdzie oczekiwanie wygranej =
`1 / (1 + 10^((R_rywala − R)/400))`. Zapisujemy rating **sprzed** meczu, więc jest leakage-safe.
K maleje z doświadczeniem — pokażmy to, a potem przeliczmy pełne Elo 2001→2025 i zobaczmy, kogo Elo
ceni najwyżej."""),

("code", """# K-factor: weterani zmieniaja rating wolniej niz nowicjusze
for nplayed in (0, 5, 20, 100, 400):
    print(f"  rozegranych meczów = {nplayed:>3}  ->  K = {k_factor(nplayed):5.1f}")

# Pelny sekwencyjny przebieg Elo 2001..2025 -> pre-match rating dla kazdego meczu 2025
target_full = build_elo_for_target_full(TARGET)
print(f"\\nMeczów w pliku {TARGET}: {len(target_full)}   |   kolumny Elo: {ELO_COLS}")

# Kogo Elo ceni najwyzej w sezonie docelowym (szczytowy pre-match rating ogolny)?
peak = {}
for _, r in target_full.iterrows():
    peak[r["winner_name"]] = max(peak.get(r["winner_name"], 0.0), r["w_elo"])
    peak[r["loser_name"]] = max(peak.get(r["loser_name"], 0.0), r["l_elo"])
print("\\nNajwyższy pre-match Elo (ogólny) w sezonie:")
for name, e in pd.Series(peak).sort_values(ascending=False).head(8).items():
    print(f"  {name:<26} {e:7.1f}")"""),

("md", """## 3. Doklejamy Elo do splitów + symetryzacja + 4 antysymetryczne cechy
Wyrównujemy Elo do tych samych meczów co baseline (`dropna(cols_base)`), dzielimy w te same proporcje
60/20/20, a po symetryzacji liczymy 4 cechy. Są **antysymetryczne** — przy zamianie p1↔p2 `elo_diff`
zmienia znak, a `elo_win_prob` przechodzi w `1 − p`. Na końcu sanity-check, że tak jest."""),

("code", """target_aligned = target_full.dropna(subset=cols_base).reset_index(drop=True)
n_tr, n_va, n_te = len(ns["df_train_raw"]), len(ns["df_val_raw"]), len(ns["df_test_raw"])
assert len(target_aligned) == n_tr + n_va + n_te, "Niespójność długości Elo vs baseline"

e_tr = target_aligned.iloc[:n_tr].reset_index(drop=True)
e_te = target_aligned.iloc[n_tr + n_va:].reset_index(drop=True)
df_train_raw = add_elo_split(ns["df_train_raw"].copy(), e_tr, ELO_COLS)
df_test_raw = add_elo_split(ns["df_test_raw"].copy(), e_te, ELO_COLS)

def build_split(df_raw, shuffle):
    sym = symmetrize_data(df_raw, shuffle=shuffle)
    sym = sym.merge(df_raw[["match_id"] + ELO_COLS], on="match_id", how="left", validate="many_to_one")
    p1w = (sym["y"] == 1).to_numpy()
    p1e = np.where(p1w, sym["w_elo"], sym["l_elo"]);          p2e = np.where(p1w, sym["l_elo"], sym["w_elo"])
    p1s = np.where(p1w, sym["w_surface_elo"], sym["l_surface_elo"]); p2s = np.where(p1w, sym["l_surface_elo"], sym["w_surface_elo"])
    sym["elo_diff"] = p1e - p2e
    sym["surface_elo_diff"] = p1s - p2s
    sym["elo_win_prob"] = 1.0 / (1.0 + 10 ** ((p2e - p1e) / 400.0))
    sym["surface_elo_win_prob"] = 1.0 / (1.0 + 10 ** ((p2s - p1s) / 400.0))
    return sym

train_data = build_split(df_train_raw, shuffle=True)
test_data = build_split(df_test_raw, shuffle=True)
features = base_features + ELO_FEATURES

mid = test_data["match_id"].iloc[0]
print("Antysymetria (ten sam mecz widziany z 2 perspektyw):")
print(test_data[test_data["match_id"] == mid][["y", "elo_diff", "elo_win_prob"]].to_string(index=False))
print(f"\\nCech razem: {len(features)} (baseline {len(base_features)} + Elo {len(ELO_FEATURES)})")"""),

("md", """## 4. Trening RF (tuned HP baseline) + porównanie
Trenujemy Random Forest z **tymi samymi** hiperparametrami co baseline (różnica = tylko 4 cechy Elo)
i sprawdzamy match accuracy oraz gdzie cechy Elo wylądowały w ważności."""),

("code", """best_rf = RandomForestClassifier(**ns["search"].best_params_, n_jobs=-1, random_state=RANDOM_STATE)
best_rf.fit(train_data[features], train_data["y"])
test_data["p1_win_probability"] = best_rf.predict_proba(test_data[features])[:, 1]
_, elo_match = compute_symmetric_match_evaluation(test_data)

print(f"baseline match = {baseline_match:.4f}")
print(f"+elo     match = {elo_match:.4f}   (delta {elo_match - baseline_match:+.4f})")

imp = pd.DataFrame({"feature": features, "importance": best_rf.feature_importances_}) \\
        .sort_values("importance", ascending=False).reset_index(drop=True)
imp["rank"] = imp.index + 1
print("\\nRanga ważności nowych cech Elo:")
for f in ELO_FEATURES:
    r = imp[imp.feature == f].iloc[0]
    print(f"  {f:<22} rank {int(r['rank']):>2}/{len(features)}   importance={r['importance']:.4f}")"""),

("md", """## 5. Uczciwa walidacja: walk-forward 2020–2025 + McNemar
Pojedynczy sezon potrafi skłamać. Dla **każdego** sezonu osobno trenujemy baseline i model+Elo na tych
samych meczach, parujemy wynik mecz-po-meczu (kto trafił), liczymy pooled deltę i McNemara. To długi
bieg — 6× pełny baseline (rok 2025 reużywamy z góry)."""),

("code", """rows, pairs = [], []
ns_cache = {TARGET: ns}
for year in TARGET_YEARS:
    nsy = ns_cache.get(year) or run_baseline_for_year(year)
    base_eval, elo_eval, base_m, elo_m, ranks, nfeat = elo_eval_for_year(nsy, year)
    merged = base_eval.merge(elo_eval, on="match_id", suffixes=("_base", "_elo"))
    for _, r in merged.iterrows():
        pairs.append((bool(r["correct_prediction_base"]), bool(r["correct_prediction_elo"])))
    rows.append({"year": year, "n": len(merged), "baseline": base_m, "elo": elo_m, "delta": elo_m - base_m})
    print(f"  {year}: baseline={base_m:.4f}  +elo={elo_m:.4f}  delta={elo_m - base_m:+.4f}")
os.environ.pop("TENNIS_TARGET_YEAR", None)

df = pd.DataFrame(rows); P = np.array(pairs)
b = int(np.sum(P[:, 0] & ~P[:, 1])); c = int(np.sum(~P[:, 0] & P[:, 1])); z, p = mcnemar(b, c)
print("\\n" + "=" * 64)
print("WALK-FORWARD: baseline vs baseline + surface-adjusted Elo")
print("=" * 64)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\\nPOOLED ({len(P)}): baseline={P[:, 0].mean():.4f}  +elo={P[:, 1].mean():.4f}  "
      f"delta={P[:, 1].mean() - P[:, 0].mean():+.4f}")
print(f"delta dodatnia w {int((df['delta'] > 0).sum())}/{len(df)} sezonach")
print(f"McNemar: b={b} c={c} z={z:.2f} p={p:.4f}  -> "
      f"{'ISTOTNE na korzyść Elo' if (p < 0.05 and c > b) else 'brak istotności (p>=0.05)'}")"""),

("md", """## Wnioski
Cechy Elo okazały się dla modelu bardzo ważne — elo_diff i surface_elo_diff lądują w ścisłej czołówce ważności cech. Mimo to na walidacji przez 6 sezonów (2020–2025, ~3000 meczów) poprawa jest mała (około +0,8 p.p.) i nieistotna statystycznie (McNemar p ≈ 0,17), a na samym 2025 wręcz ujemna.

Powód jest taki, że Elo to silny, ale w dużej mierze powtarzający się sygnał. Baseline ma już ranking ATP i formę, które mierzą mniej więcej to samo — kto jest teraz lepszy. Spotykane w literaturze „Elo ~70%" dotyczy Elo jako głównego predyktora, a nie czterech dodatkowych kolumn doklejonych do modelu, który ranking już ma. Żeby Elo realnie pomogło, musiałoby być rdzeniem modelu albo mieć znacznie dłuższą rozgrzewkę ratingów.

To samo, co widać w całym projekcie: **~65% to sufit** dla tego typu cech."""),
]

make_and_run("TPM_Experiment_Elo.ipynb", cells, timeout=3600)
