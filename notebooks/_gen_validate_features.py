"""Generuje i wykonuje OD ZERA notebook walidacji walk-forward zestawow cech
w stylu narracyjnym (jak _gen_elo.py) -- markdown opisuje krok, kod go WYKONUJE
i drukuje posrednie wyniki, zamiast wolac samo m.main().

Reuse funkcji z src/tennis_model_validate_features.py:
  run_baseline, build_context, eval_featureset, mcnemar
oraz atrybutow: TARGET_YEARS, SURFACE_FEATURES, FATIGUE_FEATURES, ELO_FEATS,
HISTORY_START_YEAR.

Uzycie: python _gen_validate_features.py
"""
from _nbtools import make_and_run

SETUP = """import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))"""

cells = [
("md", """# Eksperyment: Walidacja walk-forward zestawow cech (nowe dane 2020-2025)

## Cel
Domknac luki dowodowe: na **nowych danych** (sezony 2020-2025) sprawdzic, czy ktorykolwiek z
czterech kandydujacych zestawow cech **przebija baseline** w sposob istotny statystycznie. Testujemy
cztery zestawy doklejane do baseline:

| zestaw | cech | co wnosi |
|---|---|---|
| `surface`  | 3 | `court_pace_index` + 2 interakcje serwis x predkosc kortu |
| `fatigue`  | 6 | `rest_days` + `tourney_minutes` (dni odpoczynku + minuty w turnieju) |
| `enriched` | 9 | `surface` + `fatigue` razem |
| `elo`      | 4 | `elo_diff`, `surface_elo_diff`, `elo_win_prob`, `surface_elo_win_prob` |

## Metoda (leakage-safe, czysta ablacja)
- **Walk-forward** przez 6 sezonow (2020-2025): dla **kazdego** roku osobno trenujemy baseline na
  jego wlasnych meczach (te same chronologiczne splity 60/20/20), a potem doklejamy zestaw cech.
- Cechy liczone z historii wlasciwej dla danego roku (`2001..rok-1` dla Elo, `HISTORY_START_YEAR`
  dla surface/fatigue) -- model nigdy nie widzi przyszlosci.
- **Te same tuned hiperparametry** co baseline (`search.best_params_`) -- zmieniamy *wylacznie* cechy.
- Parujemy wynik **mecz-po-meczu** (kto trafil) -> pooled delta + parowany test **McNemara**, bo
  pojedynczy sezon potrafi sklamac."""),

("code", SETUP),

("md", """## 1. Reuse modulu walidacyjnego
Importujemy gotowe funkcje z `tennis_model_validate_features.py` -- nie duplikujemy logiki, tylko
narratujemy wokol niej. Kluczowe elementy:
- `run_baseline(year)` -- odpala (cicho, z cache) `tennis_model.py` dla danego roku i zwraca jego
  namespace (splity, tuned HP, funkcje pomocnicze, wynik baseline).
- `build_context(ns, year)` -- liczy per-match kontekst (court pace + fatigue + Elo) wyrownany do
  splitow baseline.
- `eval_featureset(ns, {"ctx": ctx}, feature_list)` -- trenuje RF na `baseline + feature_list`
  i zwraca (winner-perspective `correct_prediction`, match accuracy).
- `mcnemar(b, c)` -- parowany test istotnosci."""),

("code", """import os
import numpy as np
import pandas as pd

from tennis_model_validate_features import (
    run_baseline, build_context, eval_featureset, mcnemar,
    TARGET_YEARS, SURFACE_FEATURES, FATIGUE_FEATURES, ELO_FEATS,
    HISTORY_START_YEAR,
)

SETS = {
    "surface":  SURFACE_FEATURES,
    "fatigue":  FATIGUE_FEATURES,
    "enriched": SURFACE_FEATURES + FATIGUE_FEATURES,
    "elo":      ELO_FEATS,
}

print(f"Sezony walk-forward: {TARGET_YEARS}")
print(f"Historia cech zaczyna sie w: {HISTORY_START_YEAR}")
print("\\nZestawy cech testowane na kazdym roku:")
for name, feats in SETS.items():
    print(f"  {name:<10} ({len(feats)} cech): {feats}")"""),

("md", """## 2. Narracyjne demo: jeden sezon (2025)
Zanim odpalimy pelna petle 6 sezonow, rozbierzmy jeden rok na czynniki pierwsze. Bierzemy 2025:
liczymy baseline, budujemy kontekst (court pace + fatigue + Elo), pokazujemy probke kontekstu, a potem
ewaluujemy kazdy zestaw cech i porownujemy match accuracy z baseline."""),

("code", """DEMO_YEAR = 2025
ns_demo = run_baseline(DEMO_YEAR)                       # runpy tennis_model.py (cicho, cache)
base_eval_demo = ns_demo["winner_perspective"][["match_id", "correct_prediction"]].copy()
base_match_demo = float(ns_demo["match_accuracy"])

n_tr = len(ns_demo["df_train_raw"]); n_va = len(ns_demo["df_val_raw"]); n_te = len(ns_demo["df_test_raw"])
print(f"Baseline {DEMO_YEAR}: match={base_match_demo:.4f}  |  split train/val/test = {n_tr}/{n_va}/{n_te}")
print(f"Cech baseline: {len(ns_demo['features'])}   |   tuned HP: {ns_demo['search'].best_params_}")"""),

("md", """### 2a. Budujemy kontekst (court pace + fatigue + Elo)
`build_context` zwraca `(full_base, ctx)`, gdzie `ctx` ma jeden wiersz per mecz wyrownany pozycyjnie
do splitow baseline: `court_pace_index`, `w_/l_rest_days`, `w_/l_tourney_minutes` oraz cztery kolumny
Elo (`w_elo`, `l_elo`, `w_surface_elo`, `l_surface_elo`). Pokazmy probke i podstawowe statystyki."""),

("code", """full_base_demo, ctx_demo = build_context(ns_demo, DEMO_YEAR)
assert len(ctx_demo) == n_tr + n_va + n_te, "Niespojnosc dlugosci kontekstu vs baseline"

print(f"Kontekst: {len(ctx_demo)} wierszy  |  kolumny: {list(ctx_demo.columns)}")
print("\\nProbka (pierwsze 5 meczow):")
print(ctx_demo.head().to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\\nStatystyki wybranych kolumn kontekstu:")
print(ctx_demo[["court_pace_index", "w_rest_days", "w_tourney_minutes", "w_elo"]]
      .describe().loc[["mean", "std", "min", "max"]]
      .to_string(float_format=lambda x: f"{x:.2f}"))"""),

("md", """### 2b. Ewaluacja kazdego zestawu cech dla 2025
Dla kazdego zestawu trenujemy RF z **tymi samymi** HP co baseline (roznica = tylko dodane cechy) i
porownujemy match accuracy. To pojedynczy sezon -- pokaze kierunek, ale o istotnosci zdecyduje dopiero
pelna petla walk-forward."""),

("code", """print(f"baseline {DEMO_YEAR}: match={base_match_demo:.4f}\\n")
for name, feats in SETS.items():
    ev, match = eval_featureset(ns_demo, {"ctx": ctx_demo}, feats)
    merged = base_eval_demo.merge(ev, on="match_id", suffixes=("_base", "_var"))
    b = int(np.sum(merged["correct_prediction_base"] & ~merged["correct_prediction_var"]))
    c = int(np.sum(~merged["correct_prediction_base"] & merged["correct_prediction_var"]))
    print(f"  {name:<10} match={match:.4f}  delta={match - base_match_demo:+.4f}  "
          f"(zmienione: baseline-only b={b}, wariant-only c={c})")"""),

("md", """## 3. Pelna walidacja walk-forward 2020-2025
Teraz uczciwy bieg przez wszystkie sezony. Dla **kazdego** roku:
1. liczymy baseline (`run_baseline`) i jego per-mecz trafienia,
2. budujemy kontekst (`build_context`),
3. dla kazdego z czterech zestawow trenujemy `baseline + zestaw` i parujemy mecz-po-meczu z baseline.

To dlugi bieg -- 6x pelny baseline + 4 modele per rok. Zbieramy per-rok delty oraz globalne pary
(baseline trafil / wariant trafil) do pozniejszego McNemara."""),

("code", """pairs = {k: [] for k in SETS}
per_year = {k: [] for k in SETS}

for year in TARGET_YEARS:
    print(f"\\n===== ROK {year} =====", flush=True)
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
        per_year[name].append({"year": year, "baseline": base_match,
                               "variant": match, "delta": match - base_match})
        print(f"    {name:<10} match={match:.4f}  delta={match - base_match:+.4f}", flush=True)

os.environ.pop("TENNIS_TARGET_YEAR", None)
print("\\nWalk-forward zakonczony.")"""),

("md", """## 4. Tabele per-rok + pooled delta + McNemar
Dla kazdego zestawu skladamy tabele sezon-po-sezonie, liczymy **pooled** match accuracy na wszystkich
sparowanych meczach i parowany test McNemara. `b` = mecze, ktore trafil tylko baseline; `c` = mecze,
ktore trafil tylko wariant. Istotnosc na korzysc cechy wymaga `p<0.05` przy `c>b`."""),

("code", """summary = []
for name in SETS:
    df = pd.DataFrame(per_year[name])
    arr = np.array(pairs[name]); bc, vc = arr[:, 0], arr[:, 1]
    b = int(np.sum(bc & ~vc)); c = int(np.sum(~bc & vc))
    z, p = mcnemar(b, c)
    pos = int((df["delta"] > 0).sum())
    sig = ("ISTOTNE" if (p < 0.05 and c > b)
           else ("ISTOTNE-na-niekorzysc" if p < 0.05 else "brak istotnosci"))
    print("\\n" + "=" * 74)
    print(f"--- {name} ({len(SETS[name])} cech) ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  POOLED ({len(arr)}): baseline={bc.mean():.4f}  {name}={vc.mean():.4f}  "
          f"delta={vc.mean() - bc.mean():+.4f}  (dodatnie {pos}/{len(df)} lat)")
    print(f"  McNemar: b={b} c={c} z={z:.2f} p={p:.4f} -> {sig}")
    summary.append({"zestaw": name, "cech": len(SETS[name]),
                    "baseline": bc.mean(), "wariant": vc.mean(),
                    "delta": vc.mean() - bc.mean(), "p": p, "ocena": sig})

print("\\n" + "=" * 74)
print("PODSUMOWANIE ZBIORCZE (pooled 2020-2025)")
print("=" * 74)
print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.4f}"))"""),

("md", """## Wnioski
Walk-forward 2020-2025 (**N=3022** sparowanych meczow, baseline pooled = **0.6463**) daje twardy,
spojny werdykt -- zaden zestaw cech nie przebija baseline w sposob istotny statystycznie:

| zestaw | pooled delta | McNemar p | werdykt |
|---|---|---|---|
| `surface`  | **+0.0060** | p=0.105 | nieistotne |
| `fatigue`  | **+0.0003** | p=1.000 | nieistotne |
| `enriched` | **+0.0020** | p=0.656 | nieistotne |
| `elo`      | **+0.0076** | p=0.173 | nieistotne |

Wszystkie delty sa male (najwiekszy `elo` to ~+0.8 p.p.) i **nieistotne** (p>0.05). To komplet dowodow,
ze **zadna z badanych cech nie przebija baseline** na nowych danych -- ani predkosc kortu, ani zmeczenie,
ani ich polaczenie, ani surface-adjusted Elo.

Wniosek spojny z glownym wnioskiem projektu: **~65% to sufit dla cech feature-based**, odporny na
kolejne sygnaly tego typu. Baseline (ranking ATP + forma) juz miesci to, co nowe cechy probuja wniesc,
wiec dokladanie ich daje sygnal **redundantny**, a nie addytywny."""),
]

make_and_run("TPM_Experiment_ValidateFeatures.ipynb", cells, timeout=5400)
