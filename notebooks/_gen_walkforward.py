"""Generuje i wykonuje OD ZERA notebook walidacji walk-forward w stylu
narracyjnym (jak _gen_elo.py / TPM_Experiment_SliceAware_BestOf5_v1.ipynb) --
markdown opisuje krok, kod go WYKONUJE i drukuje posrednie wyniki, zamiast
wolac samo m.main().

Uzycie: python _gen_walkforward.py
"""
from _nbtools import make_and_run

SETUP = """import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))"""

cells = [
("md", """# Eksperyment: Walidacja walk-forward (wiele sezonow) + test istotnosci (Sprint 4)

## Cel
Sprawdzic UCZCIWIE, czy cechy wzbogacone (`surface_speed` + `fatigue`) faktycznie poprawiaja model,
czy tylko **wygladaly** dobrze na jednym sezonie. Pojedynczy test set (np. 2024, ~590 meczow) ma
przedzial ufnosci ~ +/-4 p.p., wiec zysk +1-2 p.p. moze byc czystym szumem.

## Metoda (leakage-safe, parowana)
- **Walk-forward** przez 6 sezonow docelowych (2020-2025). Dla kazdego roku `Y` osobno:
  - historia = sezony `2001..Y-1` (Sackmann), nigdy z przyszlosci,
  - baseline (RF, tuned HP) trenowany na 60% `Y`, testowany na 20% `Y`,
  - model wzbogacony na **dokladnie tych samych** meczach (czysta ablacja -- zmieniamy tylko cechy).
- Zbieramy poprawnosc **mecz po meczu** dla OBU modeli -> test **parowany** (te same mecze).
- Raport: delta per sezon, delta **pooled** (wszystkie mecze razem) + orientacyjne CI, oraz
  **McNemar** (parowany test istotnosci na meczach, gdzie modele sie roznia).
- Wiele niezaleznych test setow -> wezsze CI + sprawdzenie, czy delta jest **spojna** miedzy latami."""),

("code", SETUP),

("md", """## 1. Reuse modulu walk-forward
Importujemy publiczne funkcje z `tennis_model_walkforward.py`. Kluczowe:
- `run_baseline_for_year(Y)` -- uruchamia `tennis_model.py` (cicho, runpy) dla roku `Y` i zwraca jego
  namespace: chronologiczne splity, tuned hiperparametry, funkcje pomocnicze i wynik baseline.
- `enriched_eval_for_year(ns, Y)` -- na splitach z `ns` buduje model wzbogacony i zwraca parowane
  ewaluacje (per-mecz poprawnosc) obu modeli + ich match accuracy.
- `mcnemar(b, c)` / `wilson_delta_ci(p, n)` -- testy istotnosci.

Zerkamy tez na liste sezonow i zestaw nowych cech."""),

("code", """import os
import numpy as np
import pandas as pd

from tennis_model_walkforward import (
    run_baseline_for_year, enriched_eval_for_year,
    mcnemar, wilson_delta_ci,
    TARGET_YEARS, HISTORY_START_YEAR,
    SPEED_FEATURES, FATIGUE_FEATURES, NEW_FEATURES,
)

print(f"Sezony docelowe (walk-forward): {TARGET_YEARS}")
print(f"Historia startuje od:           {HISTORY_START_YEAR}")
print(f"\\nNowe cechy ({len(NEW_FEATURES)}):")
print(f"  surface_speed ({len(SPEED_FEATURES)}): {SPEED_FEATURES}")
print(f"  fatigue       ({len(FATIGUE_FEATURES)}): {FATIGUE_FEATURES}")"""),

("md", """## 2. Demo jednego sezonu (2025)
Zanim puscimy pelna petle, pokazmy mechanike na JEDNYM sezonie. To wlasnie taki pojedynczy wynik
potrafil kiedys "sklamac" (+2 p.p.). Liczymy baseline 2025, dokladamy cechy wzbogacone na tych samych
meczach i porownujemy match accuracy. Zwracana ramka ewaluacji ma jeden wiersz na mecz z kolumnami
`match_id` i `correct_prediction` -- to one posluza do parowania."""),

("code", """DEMO = 2025
ns_demo = run_baseline_for_year(DEMO)                 # runpy tennis_model.py (cicho)
print(f"Baseline {DEMO}: val={ns_demo['val_acc']:.4f}  test={ns_demo['test_acc']:.4f}  "
      f"match={float(ns_demo['match_accuracy']):.4f}")
print(f"Splity {DEMO}: train={len(ns_demo['df_train_raw'])}  val={len(ns_demo['df_val_raw'])}  "
      f"test={len(ns_demo['df_test_raw'])}")

base_eval, enr_eval, base_m, enr_m = enriched_eval_for_year(ns_demo, DEMO)
print(f"\\nbaseline match = {base_m:.4f}")
print(f"enriched match = {enr_m:.4f}   (delta {enr_m - base_m:+.4f})")

print("\\nPrzyklad parowanej ewaluacji (pierwsze 6 meczow test setu):")
demo_merged = base_eval.merge(enr_eval, on="match_id", suffixes=("_base", "_enr"))
print(demo_merged.head(6).to_string(index=False))"""),

("md", """## 3. Pelna petla walk-forward 2020-2025
Powtarzamy to samo dla KAZDEGO sezonu osobno i zbieramy pary `(baseline poprawny, enriched poprawny)`
ze wszystkich meczow testowych. To dlugi bieg -- 6x pelny baseline (rok 2025 reuzywamy z demo).
Drukujemy wynik per rok na biezaco, zeby widziec, jak niespojna potrafi byc delta miedzy sezonami."""),

("code", """rows, all_pairs = [], []
ns_cache = {DEMO: ns_demo}                            # nie liczmy 2025 drugi raz
for year in TARGET_YEARS:
    ns = ns_cache.get(year) or run_baseline_for_year(year)
    base_eval, enr_eval, base_m, enr_m = enriched_eval_for_year(ns, year)
    merged = base_eval.merge(enr_eval, on="match_id", suffixes=("_base", "_enr"))
    for _, r in merged.iterrows():
        all_pairs.append((bool(r["correct_prediction_base"]), bool(r["correct_prediction_enr"])))
    rows.append({"year": year, "n_test": len(merged),
                 "baseline": base_m, "enriched": enr_m, "delta": enr_m - base_m})
    print(f"  {year}: baseline={base_m:.4f}  enriched={enr_m:.4f}  "
          f"delta={enr_m - base_m:+.4f}  (n={len(merged)})", flush=True)
os.environ.pop("TENNIS_TARGET_YEAR", None)

df = pd.DataFrame(rows)
print("\\nDelta per sezon:")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\\nDelta dodatnia w {int((df['delta'] > 0).sum())}/{len(df)} sezonach")"""),

("md", """## 4. Pooled accuracy + McNemar (parowany test istotnosci)
Laczymy wszystkie mecze ze wszystkich sezonow w jeden duzy zbior (N ~ 3000) -> ciasniejsze CI niz
pojedynczy sezon. Liczymy pooled deltę oraz **McNemar**: `b` = baseline trafil, enriched nie;
`c` = odwrotnie. Istotnosc na korzysc enriched wymaga `p < 0.05` ORAZ `c > b`."""),

("code", """pairs = np.array(all_pairs)
base_correct, enr_correct = pairs[:, 0], pairs[:, 1]
N = len(pairs)
pooled_base, pooled_enr = float(base_correct.mean()), float(enr_correct.mean())
pooled_delta = pooled_enr - pooled_base

b = int(np.sum(base_correct & ~enr_correct))
c = int(np.sum(~base_correct & enr_correct))
z, pval = mcnemar(b, c)

print("=" * 72)
print("WALK-FORWARD: baseline vs enriched (surface_speed + fatigue)")
print("=" * 72)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("-" * 72)
print(f"POOLED ({N} meczow): baseline={pooled_base:.4f}  enriched={pooled_enr:.4f}  "
      f"delta={pooled_delta:+.4f}")
print(f"  Orientacyjne 95% CI (pojedyncza proporcja): +/-{wilson_delta_ci(pooled_enr, N):.4f}")
print(f"  Spojnosc: delta dodatnia w {int((df['delta'] > 0).sum())}/{len(df)} sezonach")
print()
print(f"McNemar (parowany): b(base+,enr-)={b}  c(base-,enr+)={c}  z={z:.2f}  p(two-sided)={pval:.4f}")
if pval < 0.05 and c > b:
    print("  => Roznica ISTOTNA statystycznie (p<0.05) na korzysc enriched.")
elif pval < 0.05 and b > c:
    print("  => Roznica istotna, ale na NIEKORZYSC enriched.")
else:
    print("  => Brak istotnosci (p>=0.05): zysk w granicach szumu.")"""),

("md", """## Wnioski

**Realny wynik walk-forward 2020-2025 (N=3022), cechy enriched = surface_speed + fatigue:**
- POOLED: baseline = **0.6463**, enriched = **0.6482**, **delta = +0.0020** (~+0.2 p.p.).
- McNemar: `b=60`, `c=66`, `z=0.45`, `p=0.6560` -> **brak istotnosci** (p >> 0.05).
- Delta dodatnia tylko w **3/6 sezonach** -- znak zmienny, brak spojnego trendu.

**Kluczowa lekcja metodologiczna.** Pozorny zysk z pojedynczego sezonu (kiedys nawet **+2 p.p.**)
**znika**, gdy testujemy na wielu sezonach. Pojedynczy test set (~590 meczow) ma CI ~ +/-4 p.p., wiec
zysk +1-2 p.p. miescil sie w szumie -- co potwierdza zarowno pooled delta bliska zera, jak i parowany
McNemar (`p=0.6560`). Cechy `surface_speed` + `fatigue` sa wiec w praktyce **neutralne**: ani nie
szkodza, ani nie pomagaja w sposob istotny statystycznie.

To spojne z glownym wnioskiem projektu: **~65% to sufit dla cech feature-based**, odporny na kolejne
sygnaly tego typu. Walidacja walk-forward + test parowany to wlasciwe narzedzie, by nie dac sie
zwiesc pojedynczemu "szczesliwemu" sezonowi."""),
]

make_and_run("TPM_Experiment_WalkForward.ipynb", cells, timeout=5400)
