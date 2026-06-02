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
("md", """# Eksperyment: Uczciwa walidacja wariantów slice-aware (walk-forward + McNemar)

## Cel
Warianty **slice-aware** (`sliceaware`, `bestof5_v1`, `qfserve_v3`) były dotąd oceniane
**tylko na pojedynczym teście** (sezon 2024/2025). Tam dawały spektakularne wyniki:
`bestof5_v1` raportował **+2.37 p.p.**, `qfserve_v3` **+2.20 p.p.** match accuracy nad baseline.
Ale jeden sezon potrafi skłamać. Tu sprawdzamy te warianty **uczciwie**.

## Metoda (walk-forward + test parowany)
- **Walk-forward** przez 6 sezonów (2020-2025): dla *każdego* roku osobno trenujemy baseline
  i każdy wariant na **identycznych** meczach (te same splity, ten sam seed).
- Baseline liczy się **raz na rok** i jest cache'owany (monkey-patch `runpy.run_path`), a 3 warianty
  go reużywają -- inaczej 4x pełny baseline na rok byłoby marnotrawstwem.
- **Test parowany McNemar**: porównujemy mecz-po-meczu (po `match_id`), kto trafił baseline vs wariant.
  Liczymy `b` = baseline trafił & wariant nie, `c` = wariant trafił & baseline nie. McNemar patrzy
  tylko na te *niezgodne* pary -- to właściwy test, bo pojedyncza pooled delta nie mówi nic o istotności.
- Pooled po wszystkich latach (N~3022 meczów) -> jeden wynik per wariant: pooled delta + p-value."""),

("code", SETUP),

("md", """## 1. Reuse modułu walidacji (bez `main()`)
Importujemy gotowe prymitywy z `tennis_model_validate_variants.py` zamiast duplikować logikę:

- `execute_script(name)` -- uruchamia skrypt przez (zmonkeypatchowane) `runpy.run_path` i zwraca jego
  namespace; baseline jest cache'owany per rok.
- `eval_frame(ns)` -- wyciąga z namespace'u `winner_perspective[["match_id", "correct_prediction"]]`
  (jeden wiersz na mecz: czy model trafił zwycięzcę).
- `mcnemar(b, c)` -- statystyka z poprawką ciągłości + dwustronne p (`erfc`).
- `reset_baseline_cache()` -- czyści cache baseline na początku każdego roku.
- `TARGET_YEARS`, `VARIANTS` -- definicja walk-forward i mapowanie wariant->skrypt.

**Uwaga o leakage/spójności:** każdy skrypt sam robi chronologiczny split 60/20/20 i `RANDOM_STATE=42`;
warianty importują namespace baseline (te same mecze, te same tuned HP), więc porównanie jest czystą
ablacją -- zmieniamy tylko cechy."""),

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
print(f"\\nBaseline (wspólny rdzeń): tennis_model.py  |  metryka: match_accuracy")"""),

("md", """## 2. Sanity-check McNemara
Zanim ruszymy ciężki bieg, sprawdźmy intuicję testu na ręcznych liczbach. McNemar patrzy WYŁĄCZNIE na
niezgodne pary (`b` vs `c`); zgodne (oba trafiły / oba spudłowały) nie niosą informacji o różnicy.
- duża, jednostronna przewaga (`b`=10, `c`=40) -> małe p (istotne),
- symetryczna niezgoda (`b`=25, `c`=25) -> p~1 (brak różnicy),
- brak niezgodnych par (`b`=`c`=0) -> p=1 z definicji."""),

("code", """for b, c in [(10, 40), (25, 25), (0, 0), (40, 10)]:
    z, p = mcnemar(b, c)
    verdict = "ISTOTNE" if p < 0.05 else "brak istotności"
    direction = "na korzyść wariantu" if c > b else ("na niekorzyść wariantu" if b > c else "remis")
    print(f"  b={b:>2} c={c:>2}  ->  z={z:4.2f}  p={p:.4f}  [{verdict}, {direction}]")"""),

("md", """## 3. Demo: jeden sezon (2025)
Najpierw narracyjnie rozbijmy *jeden* rok, żeby pokazać dokładnie, co robi pętla. Ustawiamy rok przez
`TENNIS_TARGET_YEAR`, czyścimy cache baseline, liczymy baseline **raz**, potem każdy wariant reużywa
cache'owany baseline. Parujemy `winner_perspective` po `match_id` i pokazujemy macierz zgodności
(`b`/`c`) dla pojedynczego wariantu -- to ten sam mechanizm, który zaraz zapętlimy po latach.

To **jednorazowy, dawny rodzaj testu** -- i właśnie tu rodziły się spektakularne pojedyncze wyniki."""),

("code", """import os

demo_year = 2025
os.environ["TENNIS_TARGET_YEAR"] = str(demo_year)
reset_baseline_cache()

# baseline raz (zapelnia cache); warianty ponizej go reuzyja
base_ns = execute_script("tennis_model.py")
base_eval = eval_frame(base_ns)
base_match = float(base_ns["match_accuracy"])
print(f"[{demo_year}] baseline match={base_match:.4f}  (n={len(base_eval)} meczów)")

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
print(f"   macierz zgodności: oba_OK={both}  oba_MISS={neither}  tylko_baseline(b)={b}  tylko_wariant(c)={c}")
print(f"   McNemar (1 sezon): z={z:.2f} p={p:.4f}  -> jeden rok to za mało, by cokolwiek orzec")"""),

("md", """## 4. Walk-forward: widoczna pętla po sezonach 2020-2025
Teraz to samo dla **każdego** roku i **każdego** wariantu, akumulując pary (base_correct, var_correct)
per wariant w `pairs` oraz wyniki per rok w `per_year`. Struktura identyczna jak w module
(`pairs`, `per_year`), ale rozpisana na widoku, z drukowaniem postępów.

To długi bieg: 6 lat x (1 baseline + 3 warianty), każde to pełny trening RF. Cache sprawia, że baseline
liczy się raz na rok."""),

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
print("\\nWalk-forward zakończony.")"""),

("md", """## 5. Per rok: tabele delt
Dla każdego wariantu pokazujemy tabelę rok-po-roku (baseline vs wariant vs delta) oraz w ilu sezonach
delta była dodatnia. Tu zwykle widać, że pojedynczy "spektakularny" sezon jest wyjątkiem, a nie
regułą -- delty skaczą w obie strony."""),

("code", """for name in VARIANTS:
    df = pd.DataFrame(per_year[name])
    pos = int((df["delta"] > 0).sum())
    print(f"--- {name} ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  delta dodatnia w {pos}/{len(df)} sezonach\\n")"""),

("md", """## 6. Pooled + McNemar (werdykt)
Łączymy wszystkie mecze ze wszystkich lat i liczymy per wariant: pooled accuracy baseline vs wariant,
pooled delta oraz McNemar (`b`, `c`, `z`, `p`). To jest *właściwa* odpowiedź na pytanie "czy wariant
istotnie pobija główny model" -- pojedynczy sezon zostaje rozcieńczony w N~3022 meczach."""),

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
        verdict = "ISTOTNE na korzyść wariantu"
    elif p < 0.05 and b > c:
        verdict = "ISTOTNE na niekorzyść wariantu"
    else:
        verdict = "brak istotności (p>=0.05)"
    print(f"\\n--- {name} ---")
    print(f"  POOLED ({N} meczów): baseline={base_c.mean():.4f}  {name}={var_c.mean():.4f}  "
          f"delta={pooled_delta:+.4f}  (dodatnie sezony {pos}/{len(df)})")
    print(f"  McNemar: b={b} c={c} z={z:.2f} p={p:.4f}  -> {verdict}")"""),

("md", """## Wnioski
Walidacja przez 6 sezonów (2020–2025, ~3000 meczów, baseline 0,6463) rozwiała optymizm z pojedynczych testów:

| wariant | zmiana | McNemar p |
|---|---|---|
| sliceaware | −0,26 p.p. | 0,61 |
| bestof5_v1 | +0,56 p.p. | 0,31 |
| qfserve_v3 | −0,89 p.p. | 0,076 |

Żaden wariant nie jest istotny (p > 0,05). Dawne efektowne wyniki z jednego sezonu (bestof5 +2,37 p.p., qfserve +2,20 p.p.) rozpłynęły się w szumie, gdy zmierzyłem je uczciwie przez 6 sezonów: bestof5 jest minimalnie dodatni, ale daleki od istotności; qfserve wręcz kierunkowo gorszy; sliceaware neutralny. To dokładnie lekcja całego projektu — pojedynczy test potrafi skłamać o jakieś 2 p.p., a prawdę mówi dopiero walidacja przez wiele sezonów z testem parowanym."""),
]

make_and_run("TPM_Experiment_ValidateVariants.ipynb", cells, timeout=6000)
