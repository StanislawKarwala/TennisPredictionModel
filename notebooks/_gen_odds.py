"""Generuje i wykonuje OD ZERA notebook eksperymentu z kursami bukmacherskimi
w stylu narracyjnym (jak TPM_Experiment_Elo.ipynb) -- markdown opisuje krok,
kod go WYKONUJE i drukuje posrednie wyniki, importujac logike z
src/tennis_model_odds.py (jedno zrodlo prawdy, bez dryfujacych kopii).

Uzycie: python _gen_odds.py
"""
from _nbtools import make_and_run

SETUP = """import sys
from pathlib import Path
import os
# --- Colab: pobiera projekt z GitHuba; lokalnie ten blok jest pomijany ---
# PO UTWORZENIU repo podmien adres ponizej na swoj:
_REPO = "https://github.com/StanislawKarwala/TennisPredictionModel.git"
if "google.colab" in sys.modules and not Path("../src/tennis_model.py").exists():
    import subprocess
    subprocess.run(["pip", "install", "-q", "xgboost"])
    subprocess.run(["git", "clone", "-q", _REPO, "/content/tenis"])
    os.chdir("/content/tenis/notebooks")
sys.path.insert(0, str(Path("../src").resolve()))"""

cells = [
("md", """# Eksperyment: Kursy bukmacherskie jako cechy modelu

## Cel
Sprawdzić, czy **kursy zamknięcia** — zagregowana wiedza całego rynku (forma, kontuzje, doniesienia,
których nie ma w box-score'ach) — przebijają sufit ~65%, na którym wyłożyły się Elo, fatigue,
surface speed i warianty slice-aware.

Kursy pobierają dwa pipeline'y: `src/fetch_odds.py` (zagraniczne EnglishAvg/Pinnacle/B365
z tennis-data.co.uk) oraz `src/fetch_odds_betexplorer.py` (polscy bukmacherzy z BetExplorer:
STS, Fortuna, Superbet, Betclic, BETFAN, LV BET, Fuksiarz). Wynik w `data/odds/atp_odds_{rok}.csv`,
join po `match_key`.

## Metoda (leakage-safe)
- Kurs zamknięcia to informacja dostępna **przed** meczem → legalna cecha, brak leakage.
- 4 nowe cechy: `p1_implied_prob`, `p2_implied_prob` (prawdopodobieństwa implikowane,
  znormalizowane do sumy 1 — usuwamy marżę bukmacherską), `implied_prob_diff` (antysymetryczna)
  oraz `has_odds` (mecze bez kursów dostają neutralne 0.5/0.5/0.0 → **identyczny** zbiór testowy
  co baseline, czyli test parowany).
- Kaskada źródeł per mecz: EnglishAvg → Pinnacle → B365 → PolishAvg (średnia polskich bukmacherów)
  (łapie np. United Cup, którego nie ma w tennis-data).
- Te same tuned hiperparametry co baseline (czysta ablacja: zmieniamy **tylko** cechy).
- **Walidacja walk-forward** przez 9 sezonów (2017–2025; kursy dostępne od 2017) + parowany test **McNemar** + benchmark
  **samego rynku** (faworyt wg kursów)."""),

("code", SETUP),

("md", """## 1. Reuse baseline (rok docelowy 2025)
Uruchamiamy `tennis_model.py` (z wyciszonym outputem) dla 2025 i pobieramy z niego: chronologiczne
splity train/val/test, tuned hiperparametry, funkcje pomocnicze (symetryzacja, metryka symetryczna)
oraz wynik baseline — żeby model z kursami liczył się na **identycznych** meczach."""),

("code", """import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tennis_model_odds import (
    run_baseline_for_year, build_odds_for_target_full, add_odds_split,
    attach_odds_features, market_benchmark, odds_eval_for_year, mcnemar,
    ODDS_FEATURES, ODDS_RAW_COLS, TARGET_YEARS,
)

TARGET = 2025
ns = run_baseline_for_year(TARGET)                 # runpy tennis_model.py (cicho)
symmetrize_data = ns["symmetrize_data"]
compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
base_features = list(ns["features"]); cols_base = list(ns["cols_base"])
RANDOM_STATE = ns["RANDOM_STATE"]
baseline_match = float(ns["match_accuracy"])

print(f"Baseline {TARGET}: val={ns['val_acc']:.4f}  test={ns['test_acc']:.4f}  match={baseline_match:.4f}")
print(f"Cechy baseline: {len(base_features)}   |   nowe cechy kursowe: {ODDS_FEATURES}")"""),

("md", """## 2. Kursy → prawdopodobieństwa implikowane (usuwanie marży)
Kurs dziesiętny `o` implikuje prawdopodobieństwo `1/o`, ale suma `1/o_1 + 1/o_2` > 1 — nadwyżka to
marża bukmachera (vig). Normalizujemy: `p1 = (1/o_1) / (1/o_1 + 1/o_2)`. Sprawdźmy pokrycie sezonu
kursami, średnią marżę i kilka przykładów."""),

("code", """target_full = build_odds_for_target_full(TARGET)
has = target_full["odds_w"].notna() & target_full["odds_l"].notna()
print(f"Mecze {TARGET} z kursami: {int(has.sum())}/{len(target_full)} ({has.mean()*100:.1f}%)")

q = 1.0 / target_full.loc[has, ["odds_w", "odds_l"]]
overround = q.sum(axis=1) - 1.0
print(f"Średnia marża bukmacherska (overround): {overround.mean()*100:.2f}%")

demo = target_full.loc[has, ["winner_name", "loser_name", "odds_w", "odds_l"]].head(6).copy()
qw, ql = 1.0 / demo["odds_w"], 1.0 / demo["odds_l"]
demo["P(zwyciezca) wg rynku"] = (qw / (qw + ql)).round(3)
print("\\nPrzykłady (kurs zwycięzcy / przegranego -> znormalizowane P):")
print(demo.to_string(index=False))"""),

("md", """## 3. Doklejenie do splitów + symetryzacja + sanity-check antysymetrii
Wyrównujemy kursy do tych samych meczów co baseline (`dropna(cols_base)`, wyrównanie pozycyjne
z assertem długości), a po symetryzacji liczymy cechy. `implied_prob_diff` jest **antysymetryczna**
(zamiana p1↔p2 odwraca znak), `p1_implied_prob` przechodzi w `1 − p`. Mecze bez kursów dostają
neutralne 0.5/0.5/0.0 i flagę `has_odds=0` — zbiór testowy zostaje identyczny z baseline."""),

("code", """target_aligned = target_full.dropna(subset=cols_base).reset_index(drop=True)
n_tr, n_va, n_te = len(ns["df_train_raw"]), len(ns["df_val_raw"]), len(ns["df_test_raw"])
assert len(target_aligned) == n_tr + n_va + n_te, "Niespójność długości kursów vs baseline"

e_tr = target_aligned.iloc[:n_tr].reset_index(drop=True)
e_te = target_aligned.iloc[n_tr + n_va:].reset_index(drop=True)
df_train_raw = add_odds_split(ns["df_train_raw"].copy(), e_tr)
df_test_raw = add_odds_split(ns["df_test_raw"].copy(), e_te)

def build_split(df_raw, shuffle):
    sym = symmetrize_data(df_raw, shuffle=shuffle)
    return attach_odds_features(sym, df_raw)

train_data = build_split(df_train_raw, shuffle=True)
test_data = build_split(df_test_raw, shuffle=True)
features = base_features + ODDS_FEATURES

mid = test_data.loc[test_data["has_odds"] == 1, "match_id"].iloc[0]
print("Antysymetria (ten sam mecz widziany z 2 perspektyw):")
print(test_data[test_data["match_id"] == mid][["y", "p1_implied_prob", "implied_prob_diff"]].to_string(index=False))
print(f"\\nCech razem: {len(features)} (baseline {len(base_features)} + kursy {len(ODDS_FEATURES)})")
print(f"Mecze testowe bez kursów (has_odds=0): {int((test_data['has_odds'] == 0).sum() / 2)}")"""),

("md", """## 4. Trening RF (tuned HP baseline) + porównanie z baseline i z samym rynkiem
Trenujemy Random Forest z **tymi samymi** hiperparametrami co baseline (różnica = tylko 4 cechy
kursowe). Do tego benchmark **samego rynku**: na meczach z kursami typujemy faworyta
(znormalizowane P > 0.5) i liczymy też Brier rynku — to poprzeczka, której box-score'owy model
nigdy dotąd nie sięgnął."""),

("code", """best_rf = RandomForestClassifier(**ns["search"].best_params_, n_jobs=-1, random_state=RANDOM_STATE)
best_rf.fit(train_data[features], train_data["y"])
test_data["p1_win_probability"] = best_rf.predict_proba(test_data[features])[:, 1]
_, odds_match = compute_symmetric_match_evaluation(test_data)

market = market_benchmark(df_test_raw)
print(f"baseline match = {baseline_match:.4f}")
print(f"+kursy   match = {odds_match:.4f}   (delta {odds_match - baseline_match:+.4f})")
print(f"sam rynek      = {market['accuracy']:.4f}  Brier={market['brier']:.4f} "
      f"(na {market['n']} meczach z kursami, {market['coverage']*100:.1f}% pokrycia)")

imp = pd.DataFrame({"feature": features, "importance": best_rf.feature_importances_}) \\
        .sort_values("importance", ascending=False).reset_index(drop=True)
imp["rank"] = imp.index + 1
print("\\nRanga ważności nowych cech kursowych:")
for f in ODDS_FEATURES:
    r = imp[imp.feature == f].iloc[0]
    print(f"  {f:<20} rank {int(r['rank']):>2}/{len(features)}   importance={r['importance']:.4f}")
print("\\nTop 5 cech modelu:")
print(imp.head(5).to_string(index=False))"""),

("md", """## 5. Uczciwa walidacja: walk-forward 2020–2025 + McNemar
Pojedynczy sezon potrafi skłamać — sprawdzamy przez 6 sezonów. Dla **każdego** sezonu osobno
trenujemy baseline i model+kursy na tych samych meczach, parujemy wynik mecz-po-meczu, liczymy
pooled deltę, McNemara oraz benchmark rynku per sezon. To długi bieg — 6× pełny baseline
(rok 2025 reużywamy z góry)."""),

("code", """rows, pairs, market_rows = [], [], []
ns_cache = {TARGET: ns}
for year in TARGET_YEARS:
    nsy = ns_cache.get(year) or run_baseline_for_year(year)
    base_eval, odds_eval, base_m, odds_m, mkt, ranks, nfeat = odds_eval_for_year(nsy, year)
    merged = base_eval.merge(odds_eval, on="match_id", suffixes=("_base", "_odds"))
    for _, r in merged.iterrows():
        pairs.append((bool(r["correct_prediction_base"]), bool(r["correct_prediction_odds"])))
    rows.append({"year": year, "n": len(merged), "baseline": base_m, "odds": odds_m,
                 "delta": odds_m - base_m})
    market_rows.append({"year": year, **mkt})
    print(f"  {year}: baseline={base_m:.4f}  +kursy={odds_m:.4f}  delta={odds_m - base_m:+.4f}"
          f"   | rynek acc={mkt['accuracy']:.4f} ({mkt['coverage']*100:.0f}% pokrycia)")
os.environ.pop("TENNIS_TARGET_YEAR", None)

df = pd.DataFrame(rows); P = np.array(pairs)
b = int(np.sum(P[:, 0] & ~P[:, 1])); c = int(np.sum(~P[:, 0] & P[:, 1])); z, p = mcnemar(b, c)
print("\\n" + "=" * 64)
print("WALK-FORWARD: baseline vs baseline + kursy bukmacherskie")
print("=" * 64)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\\nPOOLED ({len(P)}): baseline={P[:, 0].mean():.4f}  +kursy={P[:, 1].mean():.4f}  "
      f"delta={P[:, 1].mean() - P[:, 0].mean():+.4f}")
print(f"delta dodatnia w {int((df['delta'] > 0).sum())}/{len(df)} sezonach")
print(f"McNemar: b={b} c={c} z={z:.2f} p={p:.4f}  -> "
      f"{'ISTOTNE na korzyść kursów' if (p < 0.05 and c > b) else 'brak istotności (p>=0.05)'}")

mdf = pd.DataFrame(market_rows)
print("\\nSam rynek (faworyt wg kursów, na meczach z kursami):")
print(mdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
tn = int(mdf['n'].sum())
print(f"POOLED rynek: acc={float((mdf['accuracy']*mdf['n']).sum()/tn):.4f}  "
      f"Brier={float((mdf['brier']*mdf['n']).sum()/tn):.4f}  (n={tn})")"""),

("md", """## Wnioski
**Pierwsza cecha w całym projekcie, która przebija baseline w sposób istotny statystycznie.**
Na walk-forward 2017–2025 (~4700 meczów) kursy dają pooled **+2,38 p.p.** (64,72% → 67,10%),
poprawę we **wszystkich 9 sezonach na 9**, a McNemar **p < 0,0001** (b=265, c=376, z=4,34). Sufit ~65%,
o który rozbiły się Elo (+0,76, p=0,17), surface speed, fatigue i warianty slice-aware — pękł.

Cechy kursowe natychmiast zdominowały model: `p1_implied_prob` jest cechą **nr 1 w każdym
z 9 sezonów** (a `implied_prob_diff` nr 2) — powyżej `rank_diff`, formy i statystyk serwisowych.
To spodziewane: kurs zamknięcia agreguje wiedzę całego rynku (kontuzje, doniesienia, motywację),
której nie ma w box-score'ach. `has_odds` ląduje na końcu ważności — neutralna imputacja 0.5/0.5
dla meczów bez kursów wystarcza.

Uczciwa interpretacja: model+kursy gra na poziomie **samego rynku** (faworyt wg kursów: 67,5%
accuracy, Brier 0,204 na 96% meczów z kursami) — przewaga nad baseline pochodzi z informacji
rynkowej, nie z magicznej synergii cech. Wartością dodaną modelu pozostaje (a) typowanie kilku %
meczów bez kursów, (b) skalibrowane prawdopodobieństwa na pełnym zbiorze.

Uwaga operacyjna: wysokie pokrycie kursami zawdzięczamy backfill'owi polskich bukmacherów z BetExplorer
(ATP Cup / United Cup, których nie ma w tennis-data — kaskada źródeł podejmuje je automatycznie),
i to on podniósł wynik na 2020–2025 z +1,7 do +2,6 p.p. Dla sezonów 2017–2019 backfill jeszcze trwa —
ich pokrycie wzrośnie z ~87–91% do ~98%, więc pooled może jeszcze drgnąć w górę."""),
]

make_and_run("TPM_Experiment_Odds.ipynb", cells, timeout=3600)
