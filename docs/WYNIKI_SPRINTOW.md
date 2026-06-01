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
`src/tennis_model.py`, `src/tennis_model_sliceaware.py`, `src/tennis_model_sliceaware_bestof5_v1.py`, `src/tennis_model_sliceaware_qfserve_v3.py`, `src/tennis_model_seedstability.py`.

---

## Sprint 2 — HistGradientBoosting ✅ ZAMKNIĘTY (wynik NEGATYWNY)

### Co zrobione
Utworzony `src/tennis_model_hgb.py` — uczciwe porównanie (ablation): te same dane, cechy, split, symetryczna metryka co baseline; jedyna zmiana to algorytm. Dwa warianty HGB: (1) cechy numeryczne jak RF, (2) natywne kategorie (surface/tourney_level jako nominalne — główna przewaga HGB nad RF).

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

**Wniosek do pracy:** to wartościowy wynik negatywny — generyczny prior „gradient boosting bije RF na danych tabelarycznych" nie obowiązuje na małych zbiorach. RF pozostaje modelem domyślnym; `tennis_model_hgb.py` zostaje jako dowód/eksperyment.

## Sprint 3 — nowe cechy (w toku)

### 3a. Surface Speed Index ✅ DZIAŁA (+1.69 p.p.) — pomysł użytkownika
Plik `src/tennis_model_surface_speed.py`. Leakage-safe: court_pace liczony tylko z historii 2018-2023.

| Model | val | test | **match** | Brier |
|---|---|---|---|---|
| baseline | 0.6297 | 0.6153 | 0.6102 | 0.2283 |
| **+ surface_speed** | **0.6339** | **0.6212** | **0.6271** | **0.2268** |
| DELTA | +0.42 | +0.59 | **+1.69 p.p.** | lepiej |

**Wszystkie 3 metryki wzrosły spójnie + lepszy Brier** — mocny sygnał (nie pojedynczy szum). Feature importance potwierdza projekt: interakcje `serve×speed` niosą wartość (`first_won_speed_diff` rank 17/44, `ace_speed_diff` rank 18/44), samo `court_pace_index` słabsze (rank 30), `is_indoor` bezużyteczne (rank 44 — redundantne, do usunięcia w finalnej wersji).

**Wniosek:** intuicja użytkownika („szybszy kort faworyzuje mocny serw") potwierdzona danymi. To pierwszy realny zysk z nowej cechy.

### 3b. Cechy zmęczenia ✅ DZIAŁA (+1.36 p.p.)
Plik `src/tennis_model_fatigue.py`. rest_days (dni od ostatniego meczu, cap 60) + tourney_minutes (skumulowane minuty w bieżącym turnieju). Leakage-safe (chronologiczny indeks).

| Model | val | test | **match** | Brier |
|---|---|---|---|---|
| baseline | 0.6297 | 0.6153 | 0.6102 | 0.2283 |
| **+ fatigue** | **0.6407** | **0.6237** | **0.6237** | **0.2269** |
| DELTA | +1.10 | +0.85 | **+1.36 p.p.** | lepiej |

Wszystkie 3 metryki w górę spójnie. `tourney_minutes_diff` rank 27/46, `rest_days_diff` rank 33/46 — obie używane.

### 3d. Model zbiorczy (surface_speed + fatigue) ✅ NAJLEPSZY (+2.03 p.p.)
Plik `src/tennis_model_enriched.py`. Łączy 3 cechy speed + 6 fatigue (bez bezużytecznego `is_indoor`).

| Model | val | test | **match** | Brier |
|---|---|---|---|---|
| baseline | 0.6297 | 0.6153 | 0.6102 | 0.2283 |
| **+ speed + fatigue** | **0.6415** | **0.6280** | **0.6305** | **0.2256** |
| DELTA | +1.19 | +1.27 | **+2.03 p.p.** | najlepszy |

**Cechy sumują się częściowo** (osobno +1.69 i +1.36 → razem +2.03; nie pełne 3.05, bo sygnały częściowo się pokrywają). **Brak curse of dimensionality** (9 cech, w przeciwieństwie do 33 w nieudanym sliceaware). Najlepszy Brier ze wszystkich eksperymentów. To kandydat na nowy model produkcyjny — **wymaga walidacji w Sprint 4**.

### 3c. EWMA (recency weighting) ⚠️ SŁABY/NIESPÓJNY (+0.68 p.p., test płaski)
Plik `src/tennis_model_ewma_ablation.py` (α=0.18, half-life ~3.5 meczu). Inkrementalna EWMA formy/serwisu/surface_form przez chronologię 2018-2024.

| Model | val | test | **match** | Brier |
|---|---|---|---|---|
| baseline (SMA + okno 365d) | 0.6297 | 0.6153 | 0.6102 | 0.2283 |
| EWMA | 0.6458 | 0.6153 | 0.6169 | 0.2276 |
| DELTA | +1.61 | **0.00** | +0.68 | minimalnie |

**Niespójny**: val mocno w górę, test PŁASKI, match tylko +0.68. To czerwona flaga (val rośnie, ale nie generalizuje). Potwierdza hipotezę: **A3 (okno 365 dni) już wyłapał większość zysku z recency**. EWMA NIE włączony do modelu zbiorczego (dodawałby szum bez pewnego zysku).

---

## PODSUMOWANIE Sprint 3
| Cecha | match delta | spójność (val/test/match) | werdykt |
|---|---|---|---|
| surface_speed | +1.69 | ✅ wszystkie w górę | przyjęta |
| fatigue | +1.36 | ✅ wszystkie w górę | przyjęta |
| **zbiorczy (speed+fatigue)** | **+2.03** | ✅ wszystkie w górę | **kandydat produkcyjny** |
| EWMA | +0.68 | ⚠️ test płaski | odrzucona |

Model produkcyjny do walidacji w Sprint 4: **baseline + surface_speed + fatigue** (`tennis_model_enriched.py`), match 63.05%.

## Sprint 4 — walk-forward + ensemble ✅ (wynik KRYTYCZNY)

### Walidacja walk-forward: pozorny zysk +2.03 p.p. NIE GENERALIZUJE
Plik `src/tennis_model_walkforward.py`. Baseline sparametryzowany przez `TENNIS_TARGET_YEAR`. Dla 4 lat docelowych: baseline vs enriched (surface_speed + fatigue) na tych samych meczach + test parowany McNemar.

| Rok | n | baseline | enriched | delta |
|---|---|---|---|---|
| 2021 | 522 | 0.6724 | 0.6667 | **−0.57** |
| 2022 | 547 | 0.6709 | 0.6728 | +0.18 |
| 2023 | 561 | 0.6399 | 0.6257 | **−1.43** |
| 2024 | 590 | 0.6102 | 0.6305 | +2.03 |
| **POOLED** | **2220** | **0.6473** | **0.6482** | **+0.09** |

**McNemar: b=59, c=61, z=0.09, p=0.93 → ZERO istotności.** Delta dodatnia tylko w 2/4 lat.

**Walidacja maszynerii:** walk-forward dla 2024 daje +0.0203 — identycznie jak samodzielny eksperyment enriched. To dowód, że pipeline jest poprawny; +2.03 p.p. było realne DLA 2024, ale to był **szum jednego sezonu**.

### ⚠️ NAJWAŻNIEJSZY WNIOSEK CAŁEGO CYKLU
**Pozorny zysk +2.03 p.p. (surface speed + fatigue) okazał się szumem.** Na pojedynczym teście 2024 wyglądał świetnie i spójnie (val/test/match wszystkie w górę), ale walk-forward przez 4 sezony pokazał pooled +0.09 p.p. (zero) i p=0.93.

To jest dokładnie powód, dla którego Sprint 4 był konieczny — i jest to **cenny wynik metodologiczny do pracy**: na małych zbiorach (~600 meczów/sezon, CI ±4 p.p.) pojedynczy test set potrafi pokazać przekonujący, ale nieprawdziwy zysk. Walk-forward / wieloletnia walidacja jest niezbędna.

Baseline RF (~64.7% pooled match accuracy) jest prawdopodobnie blisko sufitu dla cech feature-based bez danych zewnętrznych (kursy, point-by-point).

### Walidacja adwersaryjna (workflow, 4 agentów + sceptyk)
Audyt metodologii: **walk-forward jest poprawny, 0 bugów** — agent uruchomił pipeline end-to-end i odtworzył wszystkie liczby co do cyfry (pooled delta = (c−b)/N = 2/2220 = +0.0009, McNemar z=0.09). Wniosek „cechy nie generalizują" w pełni uprawniony.

### ⚠️ Naiwny baseline „wygrywa wyżej notowany" (kluczowy kontekst)
| Rok | naive (rank) | RF baseline | przewaga RF |
|---|---|---|---|
| 2021 | 65.52% | 67.24% | +1.72 |
| 2022 | 65.45% | 67.09% | +1.64 |
| 2023 | 63.10% | 63.99% | +0.89 |
| 2024 | 61.36% | 61.02% | **−0.34** |
| **POOLED** | **63.78%** | **64.73%** | **+0.95** |

Cały RF bije naiwny ranking o tylko **+0.95 p.p.** pooled — a w 2024 jest *gorszy* od rankingu. To tłumaczy „sukces" surface_speed w 2024 (odzyskiwał sygnał, który RF tracił względem rankingu). **Zmienność accuracy między latami (61-67%) wynika ze składu meczów (faworyci vs upsety), nie z cech** — co potwierdza analiza literaturowa.

### Sufit literaturowy (zweryfikowany research)
- Akademickie modele feature-based: **64-67%** (jesteśmy w paśmie).
- Sufit bez kursów: **~70%** (surface-adjusted Elo, FiveThirtyEight).
- Z kursami bukmacherskimi: **~72-76%** (implied probability).
- Ponad ~65% podnoszą **TYLKO**: (a) implied prob z kursów, (b) surface-adjusted Elo. Kolejne cechy serwis/forma to ślepa uliczka — nasz p=0.93 zgadza się z konsensusem.
- Liczby 75%+ z blogów dotyczą pojedynczych Grand Slamów (więcej faworytów), NIE pełnego sezonu.

### Decyzja: co scalić do `tennis_model.py`
| Zmiana | typ | rekomendacja |
|---|---|---|
| A1 fix tournament_path | bugfix poprawności | **MERGE** (już scalone) |
| A2 symetryczna metryka | bugfix poprawności | **MERGE** (już scalone) |
| A3 okno 365 dni | korekta poprawności | **MERGE** (już scalone) |
| C2 neg_log_loss refit | poprawne kryterium probabilistyczne | **MERGE z caveatem** (już scalone) |
| Sprint 3 (surface/fatigue/EWMA) | nowe cechy, brak robust zysku | **zostaw jako eksperyment** (NIE importowane do main) |

### Salvage (wąskie warianty) ✅ DEFINITYWNIE: brak sygnału
Plik `src/tennis_model_salvage.py`, log `logs/salvage_run.log`. Test 4 wariantów na identycznych meczach (parowanie) przez 4 lata.

| Wariant | pooled delta | McNemar p | lata dodatnie |
|---|---|---|---|
| full (9 cech) | +0.0009 | 0.93 | 2/4 |
| speed3 (3) | +0.0032 | 0.51 | 2/4 |
| narrow2 (2) | **+0.0041** | 0.37 | 3/4 |
| single1 (1) | +0.0023 | 0.64 | 2/4 |

**Żaden wariant nie jest istotny statystycznie** (p > 0.37). Wąskie zestawy minimalnie lepsze od pełnego (potwierdza hipotezę „za dużo cech rozcieńcza"), ale nawet najlepszy (narrow2 = 2 interakcje serve×speed) daje +0.41 p.p. z p=0.37. Wszystkie warianty: ujemna korelacja delta-vs-baseline (−0.50 do −0.97) → „pomagają gdy baseline słaby", ale to regresja do średniej / artefakt 4 punktów, nie mechanizm (nie wiadomo z góry, który rok będzie trudny → niewykorzystywalne bez leakage).

**Werdykt końcowy:** cechy surface_speed/fatigue — w żadnej konfiguracji — nie dają robust zysku. Zgodne z sufitem literaturowym.
