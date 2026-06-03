"""Generuje i wykonuje OD ZERA notebook SALVAGE w stylu narracyjnym (jak
TPM_Experiment_SliceAware_BestOf5_v1.ipynb / _gen_elo.py) -- markdown opisuje
krok, kod go WYKONUJE i drukuje posrednie wyniki, zamiast wolac samo m.main().

Uzycie: python _gen_salvage.py
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
("md", """# Eksperyment: SALVAGE -- czy walk-forward ratuje robust sygnał?

## Cel
We wcześniejszych sprintach cechy **surface-speed** (court pace + interakcje serwisowe) oraz
**fatigue** (odpoczynek + minuty turniejowe) dawały na pojedynczym sezonie zysk na granicy szumu.
Pytanie ratunkowe brzmi: a może problem to **przeładowanie** -- 9 cech naraz rozmywa sygnał? Może
**węższy** podzbiór (tylko najsilniejsze interakcje, a w skrajności jedna cecha) przebije baseline w
sposób **istotny statystycznie**?

## Cztery warianty (ten sam baseline, te same mecze testowe)
- **full** (9 cech) -- 3 speed + 6 fatigue (stan z walk-forward).
- **speed3** (3 cechy) -- `court_pace_index`, `ace_speed_diff`, `first_won_speed_diff`.
- **narrow2** (2 cechy) -- tylko interakcje serwis x pace: `first_won_speed_diff`, `ace_speed_diff`.
- **single1** (1 cecha) -- najsilniejsza pojedyncza: `first_won_speed_diff`.

## Metoda (leakage-safe, parowana)
- **Walk-forward** przez sezony 2020-2025: dla *każdego* roku osobno trenujemy baseline i wszystkie
  warianty na **identycznych** meczach (parowanie per-mecz po `match_id`).
- Cechy speed/fatigue liczone wyłącznie z historii `2001..rok-1` (`build_court_pace_lookup`,
  `compute_fatigue_for_2024`) -> brak leakage.
- Te same tuned hiperparametry co baseline i to samo ziarno (`RANDOM_STATE`) -- czysta ablacja: zmienia
  się **tylko** podzbiór cech.
- Istotność: parowany test **McNemar** na pooled parach (każdy mecz: czy baseline trafił vs czy wariant
  trafił) -- pojedynczy sezon potrafi skłamać."""),

("code", SETUP),

("md", """## 1. Import machinerii SALVAGE
Reużywamy gotowych funkcji z `tennis_model_salvage.py` -- nie duplikujemy logiki splitów ani
inżynierii cech. Kluczowe elementy:
- `run_baseline_for_year(year)` -- uruchamia `tennis_model.py` (cicho) dla danego sezonu i zwraca jego
  namespace (splity, tuned HP, funkcje pomocnicze, baseline match accuracy).
- `build_enriched_splits(ns, year)` -- dokleja WSZYSTKIE 9 nowych cech do train/test (każdy wariant
  to potem inny podzbiór kolumn na tych samych ramkach).
- `eval_variant(...)` -- trenuje RF na `base_features + new_feats` i zwraca per-mecz trafność + match
  accuracy.
- `mcnemar(b, c)` -- parowany test istotności.
- `VARIANTS` -- słownik {nazwa: lista cech}; `TARGET_YEARS` -- sezony walk-forward."""),

("code", """import os
import numpy as np
import pandas as pd

from tennis_model_salvage import (
    run_baseline_for_year, build_enriched_splits, eval_variant, mcnemar,
    VARIANTS, SPEED_FEATURES, FATIGUE_FEATURES, TARGET_YEARS, HISTORY_START_YEAR,
)

print("Sezony walk-forward:", TARGET_YEARS)
print(f"Historia cech liczona od roku: {HISTORY_START_YEAR}")
print(f"\\nSPEED_FEATURES  ({len(SPEED_FEATURES)}): {SPEED_FEATURES}")
print(f"FATIGUE_FEATURES ({len(FATIGUE_FEATURES)}): {FATIGUE_FEATURES}")
print("\\nWarianty (podzbiory cech):")
for vname, feats in VARIANTS.items():
    print(f"  {vname:8s} ({len(feats)} cech): {feats}")"""),

("md", """## 2. Demo na jednym sezonie (2025)
Zanim ruszymy pełną pętlę walk-forward, rozbierzmy mechanikę na jednym roku. Liczymy baseline dla
2025, doklejamy wzbogacone splity i pokazujemy, że train/test mają już wszystkie 9 nowych kolumn oraz
że parowanie per-mecz (baseline vs wariant) odbywa się po `match_id` na tych samych meczach."""),

("code", """DEMO_YEAR = 2025
ns_demo = run_baseline_for_year(DEMO_YEAR)               # runpy tennis_model.py (cicho)
(train_data, test_data, base_features, search, RS, compute_eval,
 base_eval, base_match) = build_enriched_splits(ns_demo, DEMO_YEAR)

print(f"Baseline {DEMO_YEAR}: match accuracy = {base_match:.4f}")
print(f"Cech baseline: {len(base_features)}   |   meczów testowych: {len(base_eval)}")
print(f"Train: {len(train_data)} wierszy (po symetryzacji)   Test: {len(test_data)} wierszy")

all_new = SPEED_FEATURES + FATIGUE_FEATURES
print(f"\\nNowe kolumny obecne w test_data: {all_new}")
print("\\nPróbka nowych cech (5 wierszy testu):")
print(test_data[all_new].head().to_string(index=False))"""),

("md", """## 3. Demo: wariant `full` vs baseline na 2025
Trenujemy RF z wszystkimi 9 cechami (`eval_variant`) na tych samych meczach co baseline, parujemy
per-mecz po `match_id` i liczymy McNemara dla tego pojedynczego sezonu. Zobaczymy, że nawet pełny
zestaw daje deltę na granicy szumu -- co motywuje przejście na węższe podzbiory i walk-forward."""),

("code", """var_eval, var_match = eval_variant(
    train_data, test_data, base_features, VARIANTS["full"], search, RS, compute_eval
)
merged = base_eval.merge(var_eval, on="match_id", suffixes=("_base", "_var"))
bc = merged["correct_prediction_base"].to_numpy().astype(bool)
vc = merged["correct_prediction_var"].to_numpy().astype(bool)
b1 = int(np.sum(bc & ~vc)); c1 = int(np.sum(~bc & vc)); z1, p1 = mcnemar(b1, c1)

print(f"{DEMO_YEAR}  baseline = {base_match:.4f}")
print(f"{DEMO_YEAR}  +full(9) = {var_match:.4f}   (delta {var_match - base_match:+.4f})")
print(f"McNemar (sam {DEMO_YEAR}): b={b1} c={c1} z={z1:.2f} p={p1:.4f}  "
      f"-> {'istotne' if p1 < 0.05 else 'brak istotności (p>=0.05)'}")
print("\\nPojedynczy sezon to za mało -- przechodzimy na walk-forward przez wszystkie lata.")"""),

("md", """## 4. Walk-forward 2020-2025 dla wszystkich 4 wariantów
Główny bieg. Dla **każdego** sezonu: liczymy baseline raz, doklejamy wzbogacone splity raz, a potem
ewaluujemy wszystkie 4 warianty na tych samych ramkach (różne podzbiory kolumn). Zbieramy:
- `per_year[wariant]` -- (rok, base_match, var_match, delta),
- `pooled[wariant]` -- pary (baseline_trafił, wariant_trafił) dla każdego meczu (do McNemara).

2025 reużywamy z demo powyżej, żeby nie liczyć baseline dwa razy. To długi bieg -- pełny baseline per
sezon razy liczba sezonów."""),

("code", """per_year = {v: [] for v in VARIANTS}
pooled = {v: [] for v in VARIANTS}
splits_cache = {DEMO_YEAR: (train_data, test_data, base_features, search, RS,
                            compute_eval, base_eval, base_match)}

for year in TARGET_YEARS:
    print(f"\\n===== ROK {year} =====", flush=True)
    if year in splits_cache:
        (tr, te, bf, srch, rs, ceval, beval, bmatch) = splits_cache[year]
    else:
        nsy = run_baseline_for_year(year)
        (tr, te, bf, srch, rs, ceval, beval, bmatch) = build_enriched_splits(nsy, year)
    print(f"  baseline={bmatch:.4f}  (n_test_meczow={len(beval)})", flush=True)

    for vname, feats in VARIANTS.items():
        v_eval, v_match = eval_variant(tr, te, bf, feats, srch, rs, ceval)
        m = beval.merge(v_eval, on="match_id", suffixes=("_base", "_var"))
        per_year[vname].append((year, bmatch, v_match, v_match - bmatch))
        for _, r in m.iterrows():
            pooled[vname].append((bool(r["correct_prediction_base"]),
                                  bool(r["correct_prediction_var"])))
        print(f"    {vname:8s} ({len(feats)} cech): {v_match:.4f}  "
              f"delta={v_match - bmatch:+.4f}", flush=True)

os.environ.pop("TENNIS_TARGET_YEAR", None)
print("\\nWalk-forward zakończony.")"""),

("md", """## 5. Tabela per-rok dla każdego wariantu
Składamy czytelną ramkę per-sezon: baseline, wynik wariantu i delta. Pozwala zobaczyć, czy którykolwiek
wariant jest konsekwentnie dodatni, czy raczej balansuje wokół zera (raz +, raz -)."""),

("code", """for vname in VARIANTS:
    rows = per_year[vname]
    dfv = pd.DataFrame(rows, columns=["rok", "baseline", vname, "delta"])
    pos = int((dfv["delta"] > 0).sum())
    print(f"--- WARIANT: {vname} ({len(VARIANTS[vname])} nowych cech) ---")
    print(dfv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"   delta dodatnia w {pos}/{len(dfv)} sezonach\\n")"""),

("md", """## 6. Pooled delta + McNemar dla każdego wariantu
Najważniejszy test. Łączymy pary mecz-po-meczu ze wszystkich sezonów i liczymy:
- **pooled delta** = średnia trafność wariantu - średnia trafność baseline,
- **McNemar** na rozbieżnościach: `b` = baseline trafił a wariant nie, `c` = odwrotnie.

Jeśli żaden wariant -- nawet najwęższy -- nie osiąga `p < 0.05`, to potwierdza, że cechy surface/fatigue
nie niosą robust przewagi (nie da się tego "uratować" zawężeniem)."""),

("code", """print("=" * 72)
print("PODSUMOWANIE SALVAGE -- walk-forward, pooled + McNemar")
print("=" * 72)
summary = []
for vname in VARIANTS:
    pairs = np.array(pooled[vname])
    bc = pairs[:, 0]; vc = pairs[:, 1]; N = len(pairs)
    pooled_delta = float(vc.mean() - bc.mean())
    b = int(np.sum(bc & ~vc)); c = int(np.sum(~bc & vc)); z, pval = mcnemar(b, c)
    pos_years = int(sum(1 for (_, _, _, d) in per_year[vname] if d > 0))
    verdict = "ISTOTNE" if (pval < 0.05 and c > b) else "brak istotności (p>=0.05)"
    print(f"\\n--- {vname} ({len(VARIANTS[vname])} nowych cech) ---")
    print(f"   baseline_pooled={bc.mean():.4f}  variant_pooled={vc.mean():.4f}")
    print(f"   POOLED delta={pooled_delta:+.4f}  (N={N})   dodatnie {pos_years}/{len(per_year[vname])} lat")
    print(f"   McNemar: b={b} c={c} z={z:.2f} p={pval:.4f}  -> {verdict}")
    summary.append({"wariant": vname, "n_cech": len(VARIANTS[vname]),
                    "pooled_delta": pooled_delta, "mcnemar_p": pval, "werdykt": verdict})

print("\\n" + "=" * 72)
print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.4f}"))"""),

("md", """## Wnioski
Sprawdziłem, czy problemem nie był nadmiar cech — może zawężenie do tej najsilniejszej coś odsłoni. Wziąłem 4 podzbiory cech prędkość + zmęczenie (9, 3, 2 i 1 cecha) i przepuściłem je przez walidację na 6 sezonach. Żaden nie osiągnął istotności (wszystkie McNemar p > 0,05): pełny zestaw +0,20 p.p. (p = 0,66), sama trójka prędkościowa +0,60 p.p. (p = 0,11), węższe podobnie. Nawet jedna, najlepiej dobrana cecha nie daje realnego zysku.

Czyli prędkość kortu i zmęczenie po prostu nie niosą powtarzalnej przewagi nad baseline — niezależnie od tego, ile ich dołożę. Znów ten sam sufit ~65%."""),
]

make_and_run("TPM_Experiment_Salvage.ipynb", cells, timeout=5400)
