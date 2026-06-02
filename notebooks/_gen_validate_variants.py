"""Generuje i wykonuje OD ZERA notebook walidacji wariantow slice-aware w stylu
narracyjnym (jak _gen_elo.py) -- markdown opisuje krok, kod go WYKONUJE i drukuje
posrednie wyniki (single-season demo, widoczna petla walk-forward po latach,
tabele per rok, pooled + McNemar per wariant), zamiast wolac samo m.main().

Reuzywamy publiczne funkcje z tennis_model_validate_variants.py:
execute_script, eval_frame, mcnemar, reset_baseline_cache oraz atrybuty
TARGET_YEARS, VARIANTS. NIE wolamy m.main().

Uzycie: python _gen_validate_variants.py
"""
from _nbtools import make_and_run

SETUP = """import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))"""

cells = [
("md", """# Eksperyment: Uczciwa walidacja wariantow slice-aware (walk-forward + McNemar)

## Cel
Warianty **slice-aware** (`sliceaware`, `bestof5_v1`, `qfserve_v3`) byly dotad oceniane
**tylko na pojedynczym tescie** (sezon 2024/2025). Tam dawaly spektakularne wyniki:
`bestof5_v1` raportowal **+2.37 p.p.**, `qfserve_v3` **+2.20 p.p.** match accuracy nad baseline.
Ale jeden sezon potrafi sklamac. Tu sprawdzamy te warianty **uczciwie**.

## Metoda (walk-forward + test parowany)
- **Walk-forward** przez 6 sezonow (2020-2025): dla *kazdego* roku osobno trenujemy baseline
  i kazdy wariant na **identycznych** meczach (te same splity, ten sam seed).
- Baseline liczy sie **raz na rok** i jest cache'owany (monkey-patch `runpy.run_path`), a 3 warianty
  go reuzywaja -- inaczej 4x pelny baseline na rok byloby marnotrawstwem.
- **Test parowany McNemar**: porownujemy mecz-po-meczu (po `match_id`), kto trafil baseline vs wariant.
  Liczymy `b` = baseline trafil & wariant nie, `c` = wariant trafil & baseline nie. McNemar patrzy
  tylko na te *niezgodne* pary -- to wlasciwy test, bo pojedyncza pooled delta nie mowi nic o istotnosci.
- Pooled po wszystkich latach (N~3022 meczow) -> jeden wynik per wariant: pooled delta + p-value."""),

("code", SETUP),

("md", """## 1. Reuse modulu walidacji (bez `main()`)
Importujemy gotowe prymitywy z `tennis_model_validate_variants.py` zamiast duplikowac logike:

- `execute_script(name)` -- uruchamia skrypt przez (zmonkeypatchowane) `runpy.run_path` i zwraca jego
  namespace; baseline jest cache'owany per rok.
- `eval_frame(ns)` -- wyciaga z namespace'u `winner_perspective[["match_id", "correct_prediction"]]`
  (jeden wiersz na mecz: czy model trafil zwyciezce).
- `mcnemar(b, c)` -- statystyka z poprawka ciaglosci + dwustronne p (`erfc`).
- `reset_baseline_cache()` -- czysci cache baseline na poczatku kazdego roku.
- `TARGET_YEARS`, `VARIANTS` -- definicja walk-forward i mapowanie wariant->skrypt.

**Uwaga o leakage/spojnosci:** kazdy skrypt sam robi chronologiczny split 60/20/20 i `RANDOM_STATE=42`;
warianty importuja namespace baseline (te same mecze, te same tuned HP), wiec porownanie jest czysta
ablacja -- zmieniamy tylko cechy."""),

("code", """import numpy as np
import pandas as pd

import tennis_model_validate_variants as m
from tennis_model_validate_variants import (
    execute_script, eval_frame, mcnemar, reset_baseline_cache,
    TARGET_YEARS, VARIANTS,
)

print(f"Walk-forward lata: {TARGET_YEARS}")
print("Warianty pod testem (wariant -> skrypt):")
for name, script in VARIANTS.items():
    print(f"  {name:<12} <- {script}")
print(f"\\nBaseline (wspolny rdzen): tennis_model.py  |  metryka: match_accuracy")"""),

("md", """## 2. Sanity-check McNemara
Zanim ruszymy ciezki bieg, sprawdzmy intuicje testu na recznych liczbach. McNemar patrzy WYLACZNIE na
niezgodne pary (`b` vs `c`); zgodne (oba trafily / oba spudlowaly) nie niosa informacji o roznicy.
- duza, jednostronna przewaga (`b`=10, `c`=40) -> male p (istotne),
- symetryczna niezgoda (`b`=25, `c`=25) -> p~1 (brak roznicy),
- brak niezgodnych par (`b`=`c`=0) -> p=1 z definicji."""),

("code", """for b, c in [(10, 40), (25, 25), (0, 0), (40, 10)]:
    z, p = mcnemar(b, c)
    verdict = "ISTOTNE" if p < 0.05 else "brak istotnosci"
    direction = "na korzysc wariantu" if c > b else ("na niekorzysc wariantu" if b > c else "remis")
    print(f"  b={b:>2} c={c:>2}  ->  z={z:4.2f}  p={p:.4f}  [{verdict}, {direction}]")"""),

("md", """## 3. Demo: jeden sezon (2025)
Najpierw narracyjnie rozbijmy *jeden* rok, zeby pokazac dokladnie, co robi petla. Ustawiamy rok przez
`TENNIS_TARGET_YEAR`, czyscimy cache baseline, liczymy baseline **raz**, potem kazdy wariant reuzywa
cache'owany baseline. Parujemy `winner_perspective` po `match_id` i pokazujemy macierz zgodnosci
(`b`/`c`) dla pojedynczego wariantu -- to ten sam mechanizm, ktory zaraz zapetlimy po latach.

To **jednorazowy, dawny rodzaj testu** -- i wlasnie tu rodzily sie spektakularne pojedyncze wyniki."""),

("code", """import os

demo_year = 2025
os.environ["TENNIS_TARGET_YEAR"] = str(demo_year)
reset_baseline_cache()

# baseline raz (zapelnia cache); warianty ponizej go reuzyja
base_ns = execute_script("tennis_model.py")
base_eval = eval_frame(base_ns)
base_match = float(base_ns["match_accuracy"])
print(f"[{demo_year}] baseline match={base_match:.4f}  (n={len(base_eval)} meczow)")

# przyklad parowania dla jednego wariantu
demo_name = "bestof5_v1"
var_ns = execute_script(VARIANTS[demo_name])      # reuzywa cached baseline
var_eval = eval_frame(var_ns)
var_match = float(var_ns["match_accuracy"])
merged = base_eval.merge(var_eval, on="match_id", suffixes=("_base", "_var"))

bc = merged["correct_prediction_base"].astype(bool)
vc = merged["correct_prediction_var"].astype(bool)
b = int((bc & ~vc).sum()); c = int((~bc & vc).sum())
both = int((bc & vc).sum()); neither = int((~bc & ~vc).sum())
z, p = mcnemar(b, c)
print(f"[{demo_year}] {demo_name} match={var_match:.4f}  single-test delta={var_match-base_match:+.4f}")
print(f"   macierz zgodnosci: oba_OK={both}  oba_MISS={neither}  tylko_baseline(b)={b}  tylko_wariant(c)={c}")
print(f"   McNemar (1 sezon): z={z:.2f} p={p:.4f}  -> jeden rok to za malo, by cokolwiek orzec")"""),

("md", """## 4. Walk-forward: widoczna petla po sezonach 2020-2025
Teraz to samo dla **kazdego** roku i **kazdego** wariantu, akumulujac pary (base_correct, var_correct)
per wariant w `pairs` oraz wyniki per rok w `per_year`. Struktura identyczna jak w module
(`pairs`, `per_year`), ale rozpisana na widoku, z drukowaniem postepow.

To dlugi bieg: 6 lat x (1 baseline + 3 warianty), kazde to pelny trening RF. Cache sprawia, ze baseline
liczy sie raz na rok."""),

("code", """pairs = {name: [] for name in VARIANTS}      # (base_correct, var_correct)
per_year = {name: [] for name in VARIANTS}

for year in TARGET_YEARS:
    print(f"\\n===== ROK {year} =====", flush=True)
    os.environ["TENNIS_TARGET_YEAR"] = str(year)
    reset_baseline_cache()

    # baseline raz na rok (zapelnia cache, ktory reuzyja warianty)
    base_ns = execute_script("tennis_model.py")
    base_eval = eval_frame(base_ns)
    base_match = float(base_ns["match_accuracy"])
    print(f"  baseline match={base_match:.4f}  (n={len(base_eval)})", flush=True)

    for name, script in VARIANTS.items():
        var_ns = execute_script(script)          # reuzywa cached baseline
        var_eval = eval_frame(var_ns)
        var_match = float(var_ns["match_accuracy"])
        merged = base_eval.merge(var_eval, on="match_id", suffixes=("_base", "_var"))
        for _, r in merged.iterrows():
            pairs[name].append((bool(r["correct_prediction_base"]), bool(r["correct_prediction_var"])))
        per_year[name].append({"year": year, "baseline": base_match, "variant": var_match,
                               "delta": var_match - base_match})
        print(f"    {name:<12} match={var_match:.4f}  delta={var_match-base_match:+.4f}", flush=True)

os.environ.pop("TENNIS_TARGET_YEAR", None)
print("\\nWalk-forward zakonczony.")"""),

("md", """## 5. Per rok: tabele delt
Dla kazdego wariantu pokazujemy tabele rok-po-roku (baseline vs wariant vs delta) oraz w ilu sezonach
delta byla dodatnia. Tu zwykle widac, ze pojedynczy "spektakularny" sezon jest wyjatkiem, a nie
regula -- delty skacza w obie strony."""),

("code", """for name in VARIANTS:
    df = pd.DataFrame(per_year[name])
    pos = int((df["delta"] > 0).sum())
    print(f"--- {name} ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  delta dodatnia w {pos}/{len(df)} sezonach\\n")"""),

("md", """## 6. Pooled + McNemar (werdykt)
Laczymy wszystkie mecze ze wszystkich lat i liczymy per wariant: pooled accuracy baseline vs wariant,
pooled delta oraz McNemar (`b`, `c`, `z`, `p`). To jest *wlasciwa* odpowiedz na pytanie "czy wariant
istotnie pobija glowny model" -- pojedynczy sezon zostaje rozcienczony w N~3022 meczach."""),

("code", """print("=" * 74)
print("WALK-FORWARD: warianty slice-aware vs baseline  (pooled 2020-2025)")
print("=" * 74)
for name in VARIANTS:
    df = pd.DataFrame(per_year[name])
    arr = np.array(pairs[name])
    base_c, var_c = arr[:, 0], arr[:, 1]
    N = len(arr)
    b = int(np.sum(base_c & ~var_c))
    c = int(np.sum(~base_c & var_c))
    z, p = mcnemar(b, c)
    pooled_delta = var_c.mean() - base_c.mean()
    pos = int((df["delta"] > 0).sum())
    if p < 0.05 and c > b:
        verdict = "ISTOTNE na korzysc wariantu"
    elif p < 0.05 and b > c:
        verdict = "ISTOTNE na niekorzysc wariantu"
    else:
        verdict = "brak istotnosci (p>=0.05)"
    print(f"\\n--- {name} ---")
    print(f"  POOLED ({N} meczow): baseline={base_c.mean():.4f}  {name}={var_c.mean():.4f}  "
          f"delta={pooled_delta:+.4f}  (dodatnie sezony {pos}/{len(df)})")
    print(f"  McNemar: b={b} c={c} z={z:.2f} p={p:.4f}  -> {verdict}")"""),

("md", """## Wnioski

Walk-forward **2020-2025** (N=3022 meczow, baseline pooled match accuracy = **0.6463**) rozwial dawny
optymizm z pojedynczych testow:

| wariant | pooled delta | p (McNemar) | werdykt |
|---|---|---|---|
| `sliceaware` | **-0.0026** | 0.6097 | nieistotne |
| `bestof5_v1` | **+0.0056** | 0.3067 | nieistotne |
| `qfserve_v3` | **-0.0089** | 0.0762 | nieistotne (kierunkowo gorszy) |

**Wszystkie warianty sa statystycznie nieistotne (p > 0.05).** Dawne spektakularne wyniki z
*pojedynczego* testu (`bestof5_v1` +2.37 p.p., `qfserve_v3` +2.20 p.p.) **rozplynely sie w szumie**, gdy
zmierzylismy je uczciwie przez 6 sezonow z testem parowanym:

- `bestof5_v1` jest minimalnie dodatni (+0.56 p.p.), ale daleki od istotnosci (p=0.31) -- to nie ten
  sam wariant, ktory raportowal +2.37 p.p. na jednym sezonie; ta przewaga byla artefaktem doboru roku.
- `qfserve_v3` jest **kierunkowo gorszy** (-0.89 p.p., p=0.076) -- najblizej istotnosci, ale po zlej
  stronie: dodatkowe cechy `tourney_id`/seed/draw_size raczej przeszkadzaja niz pomagaja w skali wielu lat.
- `sliceaware` jest praktycznie neutralny (-0.26 p.p., p=0.61).

**Glowny wniosek:** zaden wariant slice-aware **nie pobija glownego modelu istotnie**. To spojne z
naczelnym ustaleniem projektu -- ~65% to sufit dla cech feature-based, a "zyski" widoczne na pojedynczym
sezonie nalezy traktowac z najwyzsza ostroznoscia, dopoki nie przejda walk-forward + testu parowanego.
Lekcja metodologiczna: **pojedynczy test moze sklamac o ~2 p.p.; dopiero pooled McNemar mowi prawde.**"""),
]

make_and_run("TPM_Experiment_ValidateVariants.ipynb", cells, timeout=6000)
