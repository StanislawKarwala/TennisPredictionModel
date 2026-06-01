# Opis pliku `tennis_model_slicecompare.py` — porównywarka wariantów

> Wszystkie pojęcia techniczne są szczegółowo wyjaśnione w `SLOWNICZEK_POJEC.md`. Tu są skrócone wytłumaczenia inline.

## Co ten plik robi w jednym zdaniu
Uruchamia wszystkie 4 modele (baseline + 3 warianty slice-aware), zbiera ich predykcje na tych samych slice'ach i pisze raport XLSX z deltami per slice per model — żeby zobaczyć, który wariant gdzie pomógł, a gdzie zaszkodził.

## Główne założenia

1. **Wspólne slice'y** — wszystkie 4 modele oceniamy NA TYCH SAMYCH testowych meczach, w tych samych slice'ach. Inaczej porównanie nie miałoby sensu (jabłka vs gruszki).

2. **Cache baseline'u** — każdy z 3 wariantów wewnętrznie wywołuje `runpy.run_path("tennis_model.py")` (przez `execute_base_pipeline_quietly`). Bez cache: baseline (50-iteracyjny RandomizedSearchCV) uruchamiałby się 4 razy w jednym wywołaniu slicecompare. Z cache: baseline odpala się RAZ, kolejne wywołania dostają ten sam namespace. Oszczędność: ~75% czasu.

3. **Target slice'y** — wyróżniamy 3 kluczowe wzorce: `best_of=5`, `round=QF`, `handedness_matchup=L-vs-R`. Te dostają flagę `is_target_slice=True` i są sortowane na górze raportu.

4. **Sanity checks** — assertion że slice setty się zgadzają między modelami (te same definicje, te same supporty). Jeśli się rozjadą, błąd — coś jest nie tak z deterministyką lub splitami.

5. **Output do `reports/outputs/`** — XLSX z dwoma arkuszami (overall_metrics, slice_comparison). Z fallbackiem timestamped jeśli plik otwarty w Excelu.

## Dlaczego takie wybory, a nie inne?

- **Czemu monkey patch `runpy.run_path`?** Bo każdy wariant ma własną funkcję `execute_base_pipeline_quietly()` — nie da się wstrzyknąć cache'u bez modyfikacji wszystkich tych funkcji. Patch globalny jest jednoosobowym fix-em.
- **Czemu XLSX a nie CSV?** Excel pozwala na freeze_panes, autofilter, autosize kolumn. Łatwiej eksplorować wyniki ręcznie.
- **Czemu inner join na slice'ach, a nie outer?** Bo każdy wariant może wygenerować lekko inny zestaw slice'ów (gdy support spadnie poniżej 5). Inner join bierze tylko slice'y wspólne dla WSZYSTKICH modeli — uczciwe porównanie.
- **Czemu `allclose` dla support, a nie `==`?** Bo niektóre wartości są float (np. po średniej). Allclose toleruje OK małe różnice numeryczne.

## Słowniczek pojęć z tego pliku

| Pojęcie | Co znaczy |
|---|---|
| **monkey patch** | Podmiana funkcji bibliotecznej w runtime. Globalnie wpływa na wszystkich callerów. |
| **cache baseline'u** | Pierwszy uruchomienie liczy się 5-10 minut (RandomizedSearchCV). Cache trzyma namespace w pamięci — kolejne wywołania są darmowe. |
| **runpy.run_path** | Funkcja Pythona uruchamiająca skrypt jako moduł i zwracająca słownik z jego zmiennymi. |
| **namespace** | Słownik wszystkich zmiennych globalnych po wykonaniu skryptu. |
| **inner join** | Join dwóch tabel zostawiający tylko wiersze obecne w OBU. |
| **outer join** | Join dwóch tabel zostawiający wszystkie wiersze, z NaN dla brakujących. |
| **target slice** | Wzorzec slice'a który jest najważniejszy dla porównania (`best_of=5`, `round=QF`, `L-vs-R`). |
| **delta vs baseline** | Różnica metryki wariantu vs baseline (np. `bo5_form_accuracy - baseline_accuracy`). |
| **freeze_panes / autosize** | Funkcje Excela: zamrażanie pierwszego wiersza, automatyczne dopasowanie szerokości kolumn. |
| **fallback output path** | Plan B gdy plik jest otwarty w Excelu (PermissionError). Zapisujemy pod timestamped nazwą. |
| **gap delta** | Zmiana gap (slice_accuracy - overall_accuracy) między wariantem a baseline. |

## Ważne metody (1:1 z kodu)

| Metoda | Co robi |
|---|---|
| `_cached_run_path(path_or_name, *args, **kwargs)` | Monkey-patch dla `runpy.run_path`. Jeśli ścieżka pokrywa się z `BASELINE_SCRIPT_PATH`, sprawdza czy w cache jest namespace — zwraca go zamiast uruchamiać skrypt. |
| `execute_model_script(script_name)` | Uruchamia dowolny wariant model script (np. `tennis_model_sliceaware.py`) z wyciszonym stdout i zwraca jego namespace. |
| `collect_slice_results(script_name)` | Dla danego skryptu: uruchamia go, buduje match_slice_frame przez `prepare_match_level_slice_frame` (z modelslice.py), liczy slice'y przez `compute_model_slices`. Zwraca (tabela slice'ów, overall match accuracy). |
| `build_model_view(slices_df, model_label)` | Zmienia nazwy kolumn slice'a żeby były prefixed nazwą modelu (np. `baseline_accuracy`, `bestof5_v1_accuracy`). Pozwala na join. |
| `build_comparison_table()` | Główna funkcja — iteruje po wszystkich 4 modelach, zbiera slice'y, mergeuje w jedną dużą tabelę. Dla każdego non-baseline modelu dorzuca kolumny delta: `{model}_accuracy_delta_vs_baseline`, `_gap_delta_vs_baseline`, `_probability_delta_vs_baseline`. Flaguje target slices. Wybiera `best_model_by_accuracy` per slice. |
| `build_overall_metrics_frame(overall_metrics)` | Buduje małą tabelę z overall match accuracy dla każdego modelu + delta vs baseline. |
| `autosize_worksheet(worksheet)` | Estetyka XLSX — dopasowuje szerokość kolumn do najdłuższej wartości. |
| `build_fallback_output_path()` | Jeśli XLSX jest otwarty w Excelu (PermissionError przy zapisie), zapisuje pod nazwą z timestampem (np. `slice_comparison_all_variants_20260514_120000.xlsx`). |
| `save_comparison_workbook(comparison, overall_metrics)` | Zapisuje XLSX z dwoma arkuszami: `overall_metrics` i `slice_comparison`. Z fallbackiem na timestamped path. |
| `print_target_slice_summary(...)` | Drukuje do konsoli: overall metryki, tabelę target slice'ów z deltami, oraz top 8 największych zysków i strat per wariant. |
| `main()` | Funkcja wejściowa — uruchamia build → save → print. |

## Ważne zmienne

| Zmienna | Co oznacza |
|---|---|
| `WORKDIR` | Katalog projektu (gdzie leży slicecompare.py — czyli `src/`). |
| `BASE_DIR` | Katalog projektu (parent WORKDIR). |
| `OUTPUT_XLSX` | Ścieżka docelowa raportu: `reports/outputs/slice_comparison_all_variants.xlsx`. |
| `BASELINE_SCRIPT_PATH` | Resolved ścieżka do `tennis_model.py` — używana w cache check. |
| `SLICE_COLUMNS` | 8 atrybutów slicingu (te same co w modelslice.py). |
| `MODELS` | Dict mapujący label → nazwa pliku. Kolejność ma znaczenie (baseline zawsze pierwszy). |
| `TARGET_SLICE_PATTERNS` | Lista wzorców target slice'ów (`best_of=5`, `round=QF`, `handedness_matchup=L-vs-R`). Substring match na `slice_definition`. |
| `_baseline_namespace_cache` | Globalna zmienna cache — None na początku, ustawiana przy pierwszym wywołaniu baseline. |
| `_original_run_path` | Backup oryginalnej `runpy.run_path` — żeby nie zacierać własnego patcha gdy ścieżka NIE pokrywa się z baseline. |
| `comparison` | Główna tabela wynikowa — wszystkie slice'y × wszystkie modele × wszystkie metryki (accuracy, gap, delta, brier itd.). |
| `overall_metrics` | Dict `{model_label: match_accuracy}`. |
| `accuracy_columns` | Lista nazw kolumn accuracy per model — używana do `idxmax` żeby znaleźć najlepszy model per slice. |

## Wyniki (rzeczywiste, po uruchomieniu)

**Overall match accuracy**:
| Model | Match accuracy | Delta vs baseline |
|---|---|---|
| baseline | 61.02% | — |
| sliceaware | 60.85% | -0.17 p.p. |
| qfserve_v3 | 63.22% | +2.20 p.p. |
| **bestof5_v1** | **63.39%** | **+2.37 p.p.** |

**Top target slice gains (qfserve_v3 najmocniejszy w R128)**:
- `round=R128 × handedness_matchup=L-vs-R`: baseline 33.3% → qfserve_v3 77.8% (**+44.4 p.p.**)
- `L-vs-R × rank_gap=0-10`: baseline 42.9% → bestof5_v1 64.3% (+21.4 p.p.)
- `tourney_level=M × L-vs-R`: baseline 61.8% → bestof5_v1 79.4% (+17.6 p.p.)

**Output**: plik `reports/outputs/slice_comparison_all_variants.xlsx` z dwoma arkuszami (overall_metrics, slice_comparison).

**Wniosek**: `bestof5_v1` daje największy globalny zysk, ale `qfserve_v3` ma najwięcej spektakularnych zysków w konkretnych slice'ach. Można rozważyć ensemble — np. używać `qfserve_v3` dla R128 i `bestof5_v1` dla reszty.

## Co odpowiedzieć gdy promotor zapyta…

**Q: „Po co osobny skrypt do porównania?"**
A: Bo bez wspólnych slice'ów porównanie wariantów to jabłka vs gruszki. Każdy wariant uruchomiony osobno może wygenerować lekko inny zestaw slice'ów (gdy support spadnie poniżej 5 — slice znika z raportu). Slicecompare gwarantuje że porównujemy WSZYSTKIE modele na DOKŁADNIE TYCH SAMYCH slice'ach.

**Q: „Co to jest 'cache baseline'u' i czemu jest ważny?"**
A: Każdy wariant wewnętrznie uruchamia baseline (`tennis_model.py`) przez `runpy.run_path` żeby dostać jego dane treningowe/testowe. Bez cache: baseline z RandomizedSearchCV (50 iteracji × 5 fold-ów = 250 fit-ów RF) uruchamiałby się 4 razy w jednym slicecompare — ~30 min. Z cache: 1 raz na początku, kolejne wywołania dostają wynik z pamięci — ~10 min. Oszczędność 75%.

**Q: „Co to jest 'monkey patch' i czemu to robicie?"**
A: Monkey patch = podmiana funkcji bibliotecznej w runtime. Tu podmieniamy `runpy.run_path` na wersję cache'ującą baseline namespace. Globalnie wpływa na WSZYSTKICH callerów `runpy.run_path` w bieżącym procesie. Pragmatyczne ale ryzykowne — jeśli inny pakiet też używa `runpy.run_path`, też zostanie wpływnięty.

**Q: „Co to jest 'inner join' na slice'ach?"**
A: Z 28 możliwych slice'ów dla każdego modelu — bierzemy tylko te WSPÓLNE dla wszystkich 4 modeli. Jeśli np. slice „best_of=5 × round=R64" istnieje u baseline'u (support=8) ale znika u sliceaware (support=4 — pod progiem 5), to wypada z porównania. Tylko slice'y obecne wszędzie zostają — uczciwe „każdy ma to samo".

**Q: „Co to jest 'target slice'?"**
A: Z 28+ slice'ów wyróżniamy 3 jako TARGET — best_of=5, round=QF, handedness_matchup=L-vs-R. Te są podświetlone (flag `is_target_slice=True`) i sortowane na górze raportu, bo to one są celem wariantów slice-aware. Pozostałe slice'y to „efekt uboczny" — chcemy widzieć czy wariant nie zepsuł czegoś innego.

**Q: „Czemu 4 warianty a nie 2 albo 5?"**
A: 4 = baseline + 3 slice-aware (sliceaware shotgun, bestof5_v1 focused, qfserve_v3 rich). To są SENSOWNE warianty z naszej historii: sliceaware to baseline + 20 cech (nieudany), bo5 to focused (najlepszy), qfserve_v3 to wszystko + seed/path (drugi najlepszy). Dodanie kolejnych wariantów wymagałoby kolejnych plików `tennis_model_*.py`.

**Q: „Czemu format XLSX a nie CSV?"**
A: CSV jest plain text, łatwy do parsowania ale brzydki do oglądania. XLSX pozwala na:
- Freeze panes (pierwszy wiersz zostaje na górze gdy scrollujesz)
- Autofilter (klikalne strzałki w nagłówku — filtruj po wartości)
- Autosize kolumn (każda kolumna szeroka na najdłuższą wartość)
- Dwa arkusze w jednym pliku (overall_metrics + slice_comparison)

Promotor otwiera w Excelu i wszystko czyta od razu.

**Q: „Co robi 'fallback output path'?"**
A: Plan B na sytuację: chcesz zapisać `slice_comparison_all_variants.xlsx` ale ten plik JEST OTWARTY w Excelu (PermissionError). Zamiast crashować, zapisujemy pod nazwą z timestampem: `slice_comparison_all_variants_20260514_120000.xlsx`. Promotor widzi obie wersje — nie traci wyników.

**Q: „Co to jest 'ensemble' o którym mówisz w konkluzji?"**
A: Pomysł na przyszłość: zamiast wybierać JEDEN wariant, mieć system który dla każdego meczu wybiera najlepszy wariant. Np. „jeśli mecz to R128 z L-vs-R → użyj qfserve_v3 (+44 p.p.). Jeśli mecz to Bo5 → użyj bestof5_v1. Inaczej → bestof5_v1 (najlepszy ogólnie)". To by dało więcej niż sam najlepszy wariant. Wymaga dodatkowego pliku routing.py.
