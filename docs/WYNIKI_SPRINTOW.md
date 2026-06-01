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

## Sprint 2 — HistGradientBoosting (w toku)
_(wyniki po implementacji)_

## Sprint 3 — EWMA + zmęczenie + surface speed (oczekuje)

## Sprint 4 — walk-forward + ensemble (oczekuje)
