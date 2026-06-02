"""Generuje i wykonuje OD ZERA notebook SALVAGE w stylu narracyjnym (jak
TPM_Experiment_SliceAware_BestOf5_v1.ipynb / _gen_elo.py) -- markdown opisuje
krok, kod go WYKONUJE i drukuje posrednie wyniki, zamiast wolac samo m.main().

Uzycie: python _gen_salvage.py
"""
from _nbtools import make_and_run

SETUP = """import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))"""

cells = [
("md", """# Eksperyment: SALVAGE -- czy walk-forward ratuje robust sygnal? (Sprint 5)

## Cel
We wczesniejszych sprintach cechy **surface-speed** (court pace + interakcje serwisowe) oraz
**fatigue** (odpoczynek + minuty turniejowe) dawaly na pojedynczym sezonie zysk na granicy szumu.
Pytanie ratunkowe brzmi: a moze problem to **przeladowanie** -- 9 cech naraz rozmywa sygnal? Moze
**wezszy** podzbior (tylko najsilniejsze interakcje, a w skrajnosci jedna cecha) przebije baseline w
sposob **istotny statystycznie**?

## Cztery warianty (ten sam baseline, te same mecze testowe)
- **full** (9 cech) -- 3 speed + 6 fatigue (stan z walk-forward).
- **speed3** (3 cechy) -- `court_pace_index`, `ace_speed_diff`, `first_won_speed_diff`.
- **narrow2** (2 cechy) -- tylko interakcje serwis x pace: `first_won_speed_diff`, `ace_speed_diff`.
- **single1** (1 cecha) -- najsilniejsza pojedyncza: `first_won_speed_diff`.

## Metoda (leakage-safe, parowana)
- **Walk-forward** przez sezony 2020-2025: dla *kazdego* roku osobno trenujemy baseline i wszystkie
  warianty na **identycznych** meczach (parowanie per-mecz po `match_id`).
- Cechy speed/fatigue liczone wylacznie z historii `2001..rok-1` (`build_court_pace_lookup`,
  `compute_fatigue_for_2024`) -> brak leakage.
- Te same tuned hiperparametry co baseline i to samo ziarno (`RANDOM_STATE`) -- czysta ablacja: zmienia
  sie **tylko** podzbior cech.
- Istotnosc: parowany test **McNemar** na pooled parach (kazdy mecz: czy baseline trafil vs czy wariant
  trafil) -- pojedynczy sezon potrafi sklamac."""),

("code", SETUP),

("md", """## 1. Import machinerii SALVAGE
Reuzywamy gotowych funkcji z `tennis_model_salvage.py` -- nie duplikujemy logiki splitow ani
inzynierii cech. Kluczowe elementy:
- `run_baseline_for_year(year)` -- uruchamia `tennis_model.py` (cicho) dla danego sezonu i zwraca jego
  namespace (splity, tuned HP, funkcje pomocnicze, baseline match accuracy).
- `build_enriched_splits(ns, year)` -- doklejaja WSZYSTKIE 9 nowych cech do train/test (kazdy wariant
  to potem inny podzbior kolumn na tych samych ramkach).
- `eval_variant(...)` -- trenuje RF na `base_features + new_feats` i zwraca per-mecz trafnosc + match
  accuracy.
- `mcnemar(b, c)` -- parowany test istotnosci.
- `VARIANTS` -- slownik {nazwa: lista cech}; `TARGET_YEARS` -- sezony walk-forward."""),

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
Zanim ruszymy pelna petle walk-forward, rozbierzmy mechanike na jednym roku. Liczymy baseline dla
2025, doklejamy wzbogacone splity i pokazujemy, ze train/test maja juz wszystkie 9 nowych kolumn oraz
ze parowanie per-mecz (baseline vs wariant) odbywa sie po `match_id` na tych samych meczach."""),

("code", """DEMO_YEAR = 2025
ns_demo = run_baseline_for_year(DEMO_YEAR)               # runpy tennis_model.py (cicho)
(train_data, test_data, base_features, search, RS, compute_eval,
 base_eval, base_match) = build_enriched_splits(ns_demo, DEMO_YEAR)

print(f"Baseline {DEMO_YEAR}: match accuracy = {base_match:.4f}")
print(f"Cech baseline: {len(base_features)}   |   meczow testowych: {len(base_eval)}")
print(f"Train: {len(train_data)} wierszy (po symetryzacji)   Test: {len(test_data)} wierszy")

all_new = SPEED_FEATURES + FATIGUE_FEATURES
print(f"\\nNowe kolumny obecne w test_data: {all_new}")
print("\\nProbka nowych cech (5 wierszy testu):")
print(test_data[all_new].head().to_string(index=False))"""),

("md", """## 3. Demo: wariant `full` vs baseline na 2025
Trenujemy RF z wszystkimi 9 cechami (`eval_variant`) na tych samych meczach co baseline, parujemy
per-mecz po `match_id` i liczymy McNemara dla tego pojedynczego sezonu. Zobaczymy, ze nawet pelny
zestaw daje delte na granicy szumu -- co motywuje przejscie na wezsze podzbiory i walk-forward."""),

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
      f"-> {'istotne' if p1 < 0.05 else 'brak istotnosci (p>=0.05)'}")
print("\\nPojedynczy sezon to za malo -- przechodzimy na walk-forward przez wszystkie lata.")"""),

("md", """## 4. Walk-forward 2020-2025 dla wszystkich 4 wariantow
Glowny bieg. Dla **kazdego** sezonu: liczymy baseline raz, doklejamy wzbogacone splity raz, a potem
ewaluujemy wszystkie 4 warianty na tych samych ramkach (rozne podzbiory kolumn). Zbieramy:
- `per_year[wariant]` -- (rok, base_match, var_match, delta),
- `pooled[wariant]` -- pary (baseline_trafil, wariant_trafil) dla kazdego meczu (do McNemara).

2025 reuzywamy z demo powyzej, zeby nie liczyc baseline dwa razy. To dlugi bieg -- pelny baseline per
sezon razy liczba sezonow."""),

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
print("\\nWalk-forward zakonczony.")"""),

("md", """## 5. Tabela per-rok dla kazdego wariantu
Skladamy czytelna ramke per-sezon: baseline, wynik wariantu i delta. Pozwala zobaczyc, czy ktorykolwiek
wariant jest konsekwentnie dodatni, czy raczej balansuje wokol zera (raz +, raz -)."""),

("code", """for vname in VARIANTS:
    rows = per_year[vname]
    dfv = pd.DataFrame(rows, columns=["rok", "baseline", vname, "delta"])
    pos = int((dfv["delta"] > 0).sum())
    print(f"--- WARIANT: {vname} ({len(VARIANTS[vname])} nowych cech) ---")
    print(dfv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"   delta dodatnia w {pos}/{len(dfv)} sezonach\\n")"""),

("md", """## 6. Pooled delta + McNemar dla kazdego wariantu
Najwazniejszy test. Laczymy pary mecz-po-meczu ze wszystkich sezonow i liczymy:
- **pooled delta** = srednia trafnosc wariantu - srednia trafnosc baseline,
- **McNemar** na rozbieznosciach: `b` = baseline trafil a wariant nie, `c` = odwrotnie.

Jesli zaden wariant -- nawet najwezszy -- nie osiaga `p < 0.05`, to potwierdza, ze cechy surface/fatigue
nie nios robust przewagi (nie da sie tego "uratowac" zawezeniem)."""),

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
    verdict = "ISTOTNE" if (pval < 0.05 and c > b) else "brak istotnosci (p>=0.05)"
    print(f"\\n--- {vname} ({len(VARIANTS[vname])} nowych cech) ---")
    print(f"   baseline_pooled={bc.mean():.4f}  variant_pooled={vc.mean():.4f}")
    print(f"   POOLED delta={pooled_delta:+.4f}  (N={N})   dodatnie {pos_years}/{len(per_year[vname])} lat")
    print(f"   McNemar: b={b} c={c} z={z:.2f} p={pval:.4f}  -> {verdict}")
    summary.append({"wariant": vname, "n_cech": len(VARIANTS[vname]),
                    "pooled_delta": pooled_delta, "mcnemar_p": pval, "werdykt": verdict})

print("\\n" + "=" * 72)
print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.4f}"))"""),

("md", """## Wnioski
Walk-forward przez sezony **2020-2025** (pooled **N=3022** par mecz-po-meczu), 4 podzbiory cech
surface+fatigue: **full** (9), **speed3** (3), **narrow2** (2), **single1** (1).

**Wszystkie warianty maja McNemar p > 0.05** -- zaden nie osiaga istotnosci statystycznej:
- `full`  (9 cech): pooled delta ~ **+0.0020**, McNemar **p ~ 0.656**,
- `speed3` (3 cechy): pooled delta ~ **+0.0060**, McNemar **p ~ 0.105**,
- `narrow2` (2 cechy) i `single1` (1 cecha): rowniez p > 0.05.

**Nawet najwezszy podzbior (1 cecha) nie daje istotnego zysku.** Hipoteza ratunkowa -- ze problemem
bylo przeladowanie 9 cechami, a zawezenie do najsilniejszej interakcji `first_won_speed_diff` odsloni
robust sygnal -- **nie potwierdza sie**. Zawezanie liczby cech zmienia delte tylko kosmetycznie i nigdy
nie przekracza progu istotnosci.

Wniosek koncowy: cechy **surface-speed** i **fatigue** **nie nios robust, powtarzalnej przewagi** nad
baseline. To spojne z glownym wnioskiem projektu: **~65% to praktyczny sufit dla cech feature-based**,
odporny na kolejne sygnaly tego typu -- niezaleznie od tego, czy dodajemy ich wiele, czy tylko jedna,
najlepiej dobrana."""),
]

make_and_run("TPM_Experiment_Salvage.ipynb", cells, timeout=5400)
