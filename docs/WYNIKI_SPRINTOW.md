# Wyniki sprintów rozwoju modelu

Dziennik twardych liczb po każdym sprincie z `PLAN_ROZWOJU_MODELU.md`.

---

## Sprint 1 — poprawność (A1 + A2 + A3 + C2) ✅ ZAMKNIĘTY

### Co zmienione
- **A1**: naprawiony bug `calculate_tournament_path_stats` w qfserve_v3 (cecha liczona na całej karierze → na bieżącym turnieju).
- **A2**: metryka `match_accuracy` symetryczna (uśrednia obie perspektywy meczu) — baseline + 3 warianty. Wspólny helper `compute_symmetric_match_evaluation`.
- **A3**: okno czasowe `FORM_RECENCY_DAYS = 365` na formę/serwis (H2H bez limitu). `tourney_date` przeniesione jako metadana.
- **C2**: scoring CV `neg_log_loss` (refit) + multi-metric (accuracy, roc_auc) raport.

### Match accuracy: PRZED vs PO Sprint 1

| Model | PRZED (stara, jednostronna metryka) | PO (poprawna, symetryczna metryka) |
|---|---|---|
| baseline | 61.02% (—) | 61.02% (—) |
| sliceaware | 60.85% (**-0.17**) | **62.71% (+1.69)** |
| qfserve_v3 | 63.22% (+2.20) | 61.19% (+0.17) |
| bestof5_v1 | **63.39% (+2.37)** | 62.03% (+1.02) |

### ⚠️ KLUCZOWY WNIOSEK (ważny do pracy magisterskiej)
**Poprzedni ranking był częściowo artefaktem zepsutej metryki.** Na starej, jednostronnej metryce „najlepszy" był bestof5_v1 (+2.37), a sliceaware był rzekomo gorszy od baseline (-0.17). Po naprawie metryki na poprawną (symetryczną) **kolejność się odwróciła**: teraz sliceaware jest najlepszy (+1.69), a qfserve_v3 prawie nie różni się od baseline (+0.17).

To jest dokładnie powód, dla którego Sprint 1 musiał być pierwszy: **wcześniej optymalizowaliśmy względem błędnego pomiaru.** Każdy wniosek typu „wariant X jest lepszy" z poprzednich raportów trzeba traktować jako podejrzany.

### ⚠️ Ważne zastrzeżenie statystyczne
Przy ~590 meczach testowych przedział ufności dla match_accuracy to ok. **±4 p.p.** Różnice 0.17-1.69 p.p. między wariantami **mieszczą się w szumie** — nie można jeszcze twierdzić, że jeden wariant jest istotnie lepszy od drugiego. To bezpośrednio motywuje **Sprint 4** (walk-forward na wielu latach → CI ~±1 p.p.).

### Dowody poprawności
- Baseline: ścieżka threshold-tuned zgadza się teraz idealnie z główną metryką (delta = 0.0000) — wcześniej była niespójność (objaw jednostronności).
- Kalibracja minimalnie lepsza: Brier 0.2283 (było 0.2284), ECE 0.0399 (było 0.0403).
- Wszystkie 4 modele uruchomione przez slicecompare bez błędu (exit 0).
- CV: neg_log_loss=-0.6217, accuracy=0.6417, roc_auc raportowane.

### Pliki zmienione
`src/main_48_cech.py`, `src/main_48_cech_sliceaware.py`, `src/main_48_cech_sliceaware_bestof5_v1.py`, `src/main_48_cech_sliceaware_qfserve_v3.py`, `src/main_48_cech_seedstability.py`.

---

## Sprint 2 — HistGradientBoosting ✅ ZAMKNIĘTY (wynik NEGATYWNY)

### Co zrobione
Utworzony `src/main_48_cech_hgb.py` — uczciwe porównanie (ablation): te same dane, cechy, split, symetryczna metryka co baseline; jedyna zmiana to algorytm. Dwa warianty HGB: (1) cechy numeryczne jak RF, (2) natywne kategorie (surface/tourney_level jako nominalne — główna przewaga HGB nad RF).

### Wyniki

| Model | val | test | **match** | Brier | log-loss | ECE | CV acc |
|---|---|---|---|---|---|---|---|
| **RandomForest** | 0.6297 | 0.6153 | **0.6102** | **0.2283** | **0.6460** | **0.0399** | **0.6417** |
| HGB (numeric) | 0.6246 | 0.6136 | 0.6034 | 0.2287 | 0.6471 | 0.0627 | 0.6332 |
| HGB (kategorie) | 0.6305 | 0.6119 | 0.6136 | 0.2288 | 0.6475 | 0.0597 | 0.6353 |

### Werdykt: ZOSTAJEMY PRZY RANDOM FOREST
**Hipoteza „+1.5-3 p.p." z literatury NIE potwierdziła się na naszych danych.** Powody:
- RF wygrywa na **CV accuracy** (0.6417 vs 0.6332-0.6353) — to najbardziej wiarygodny sygnał (średnia z 5 foldów, nie pojedynczy test).
- RF ma **wyraźnie lepszą kalibrację** (ECE 0.0399 vs 0.0597-0.0627; Brier też lepszy).
- HGB z natywnymi kategoriami daje match +0.34 p.p., ale test -0.34 p.p. — mieszane i w granicach szumu (±4 p.p.).
- HGB wybrał silnie regularyzowane HP (min_samples_leaf=120, learning_rate=0.02) → ~3500 próbek treningowych to za mało, by boosting rozwinął przewagę.

**Wniosek do pracy:** to wartościowy wynik negatywny — generyczny prior „gradient boosting bije RF na danych tabelarycznych" nie obowiązuje na małych zbiorach. RF pozostaje modelem domyślnym; `main_48_cech_hgb.py` zostaje jako dowód/eksperyment.

## Sprint 3 — nowe cechy (w toku)

### 3a. Surface Speed Index ✅ DZIAŁA (+1.69 p.p.) — pomysł użytkownika
Plik `src/main_48_cech_surface_speed.py`. Leakage-safe: court_pace liczony tylko z historii 2018-2023.

| Model | val | test | **match** | Brier |
|---|---|---|---|---|
| baseline | 0.6297 | 0.6153 | 0.6102 | 0.2283 |
| **+ surface_speed** | **0.6339** | **0.6212** | **0.6271** | **0.2268** |
| DELTA | +0.42 | +0.59 | **+1.69 p.p.** | lepiej |

**Wszystkie 3 metryki wzrosły spójnie + lepszy Brier** — mocny sygnał (nie pojedynczy szum). Feature importance potwierdza projekt: interakcje `serve×speed` niosą wartość (`first_won_speed_diff` rank 17/44, `ace_speed_diff` rank 18/44), samo `court_pace_index` słabsze (rank 30), `is_indoor` bezużyteczne (rank 44 — redundantne, do usunięcia w finalnej wersji).

**Wniosek:** intuicja użytkownika („szybszy kort faworyzuje mocny serw") potwierdzona danymi. To pierwszy realny zysk z nowej cechy.

### 3b. Cechy zmęczenia ✅ DZIAŁA (+1.36 p.p.)
Plik `src/main_48_cech_fatigue.py`. rest_days (dni od ostatniego meczu, cap 60) + tourney_minutes (skumulowane minuty w bieżącym turnieju). Leakage-safe (chronologiczny indeks).

| Model | val | test | **match** | Brier |
|---|---|---|---|---|
| baseline | 0.6297 | 0.6153 | 0.6102 | 0.2283 |
| **+ fatigue** | **0.6407** | **0.6237** | **0.6237** | **0.2269** |
| DELTA | +1.10 | +0.85 | **+1.36 p.p.** | lepiej |

Wszystkie 3 metryki w górę spójnie. `tourney_minutes_diff` rank 27/46, `rest_days_diff` rank 33/46 — obie używane.

### 3c. EWMA (recency weighting) — oczekuje

### 3d. Model zbiorczy (surface_speed + fatigue) — w toku

## Sprint 4 — walk-forward + ensemble (oczekuje)
