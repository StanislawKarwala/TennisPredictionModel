# Code Review — TenisPredictionModel

Data przeglądu: 2026-06-11. Zakres: cały kod w `src/` (główny pipeline + 19 wariantów), wszystkie pliki `.md` w `docs/` i `reports/`, wszystkie notebooki w `notebooks/` (15 głównych + 3 archiwalne) oraz skrypty generujące `_gen_*.py`.

**Werdykt ogólny: rdzeń projektu jest metodologicznie zdrowy.** Nie znaleziono żadnego krytycznego data leakage w aktywnym kodzie: cechy dynamiczne liczone są wyłącznie z meczów sprzed bieżącego (expanding window z `bisect_left`, ścisłe `< cutoff`), split jest chronologiczny, symetryzacja poprawnie odwraca/neguje wszystkie cechy w lustrzanym wierszu, kalibrator jest uczony tylko na walidacji, a Elo aktualizuje ratingi dopiero PO pobraniu cech pre-match. Znalezione i naprawione błędy to klasa „mniejsza, ale realna": błędna korekta ciągłości w teście McNemara, odwrócony znak i nieaktualne liczby we wnioskach notebooków, błędna definicja flagi `is_lefty_matchup`, podwójne liczenie meczu przy zimnym starcie EWMA oraz spora liczba rozjazdów dokumentacja↔kod.

---

## 1. Znalezione i naprawione błędy logiczne

### 1.1. Kod źródłowy (`src/`)

| # | Plik | Błąd | Naprawa |
|---|------|------|---------|
| 1 | `tennis_model_elo.py`, `tennis_model_walkforward.py`, `tennis_model_salvage.py`, `tennis_model_validate_features.py`, `tennis_model_validate_variants.py` | **Test McNemara: korekta ciągłości schodziła poniżej zera.** `z = (abs(b-c)-1)/sqrt(n)` przy `b == c` daje `z = -1/sqrt(n)`, a późniejsze `abs(z)` produkuje p < 1 (np. ~0.75–0.97) zamiast poprawnego p = 1.0 dla braku różnicy. Wynik antykonserwatywny (test „widział" różnicę tam, gdzie jej nie ma). | We wszystkich 5 kopiach: `z = max(abs(b-c)-1, 0)/sqrt(n)` |
| 2 | `tennis_model_salvage.py:224` | **Zły mianownik w raporcie**: `dodatnie {pos_years}/4 lat` na sztywno, podczas gdy domyślne `TARGET_YEARS` ma 6 sezonów (2020–2025). | `/{len(rows)}` |
| 3 | `tennis_model_sliceaware.py:563`, `tennis_model_sliceaware_qfserve_v3.py:952` | **`is_lefty_matchup` liczone jako „różna ręczność"** (`winner_hand != loser_hand`), więc mecz R-vs-U (U = ręka nieznana; 111 takich wpisów w danych 2024–25) był flagowany jako matchup z leworęcznym. Cecha nie odpowiadała ani nazwie, ani slice'owi L-vs-R, w który celowała. | XOR po `'L'`: dokładnie jeden leworęczny na korcie |
| 4 | `tennis_model_ewma_ablation.py:151-155` | **Seed surface-EWMA liczył bieżący mecz podwójnie.** `form_state` aktualizowany był przed `surf_state`, więc fallback `get_surf → get_form` przy zimnym starcie pary (gracz, nawierzchnia) blendował wynik bieżącego meczu drugi raz. Nie był to leakage (cechy zapisywane wcześniej), ale obciążenie zimnego startu. | Zamiana kolejności: surface form aktualizowany przed formą ogólną |
| 5 | `tennis_model_surface_speed.py:120` | **`float(std) or 0.02` nie łapie NaN** (NaN jest truthy). Przy < 2 turniejach z wystarczającym wsparciem `std()` zwraca NaN i zatruwa cały `court_pace_index` (NaN w cechach → crash RF). Z domyślną historią nie triggeruje, z krótkim `TENNIS_HISTORY_START` — tak. | Jawny guard `np.isfinite` |
| 6 | `tennis_model_seedstability.py:61-63` | **Fallback wstawiał neg_log_loss do kolumny `cv_accuracy`.** Przy starszym baseline bez eksportu `cv_accuracy` brany był `search.best_score_`, który po zmianie na `refit='neg_log_loss'` jest ujemnym log-lossem — podsumowanie mean/std mieszałoby dwie różne metryki. Ścieżka martwa dziś, ale uzbrojona. | Fallback czyta `mean_test_accuracy` z `cv_results_`, a `best_score_` tylko przy `scoring == 'accuracy'`; inaczej NaN |
| 7 | `tennis_model_elo.py:131-135` | `add_elo_split` ignorował parametr `context_cols` (używał globalnego `ELO_COLS`) — latentny błąd API. | Funkcja używa parametru |
| 8 | `tennis_model_validate_features.py:223` | Martwe wyrażenie `{len(arr[0]) if False else len(SETS[name])}`. | Uproszczone do `len(SETS[name])` |
| 9 | `tennis_model.py` (docstring), `tennis_model_walkforward.py` (docstring), `tennis_model_surface_speed.py` (docstring) | Nagłówki opisywały stare konfiguracje (rok 2024, historia 2018–2023, lata WF 2021–2024, cechę `is_indoor` która została usunięta), gdy kod ma domyślnie TARGET_YEAR=2025, historię od 2001, WF 2020–2025. | Docstringi zaktualizowane do stanu parametrycznego |

**Zweryfikowane i NIE będące błędami** (sprawdzone na danych/kodzie):
- `np.log(rank_points)` — minimum punktów rankingowych we wszystkich 27 plikach CSV po dropna wynosi 1, więc log(0) nigdy nie występuje.
- Cechy `w_form`/`l_form`/`w_ace_rate`… na ramkach raw wyglądają jak leakage (nazwy kodują wynik), ale są to rolling staty PRE-match; symetryzacja mapuje je na p1/p2 niezależnie od wyniku.
- Statystyki serwisowe bieżącego meczu (`w_ace`, `l_svpt`…) są w `cols_base`, ale służą wyłącznie jako historia dla PÓŹNIEJSZYCH meczów — nigdy nie wchodzą do cech własnego wiersza.
- Założenie „`match_num` rośnie zgodnie z rundami w obrębie turnieju" (krytyczne dla `tourney_path_*` w qfserve_v3) — zweryfikowane empirycznie na wszystkich 141 turniejach 2025: trzyma się.

### 1.2. Notebooki — błędne lub nieaktualne wnioski (naprawione)

Mechanizm powstania błędów: skrypty `_gen_*.py` mają wnioski zahardkodowane jako tekst z wcześniejszego uruchomienia; po ponownym wykonaniu (głównie w Colab) liczby w outputach się zmieniły, a wnioski nie. Dwa przypadki to błędy merytoryczne (zły znak / teza odwrotna do własnej tabeli), reszta to stale liczby.

| Notebook | Błąd | Naprawa |
|---|---|---|
| `TPM_Experiment_Fatigue` | Wniosek: „**+0,03 p.p.**" — własny artefakt walidacyjny pokazuje **−0,03 p.p.** (znak odwrócony); „model z nich korzysta" przy rangach ważności 28–39/46 | Poprawiony znak; doprecyzowana niska waga cech |
| `TPM_Experiment_HGB` | „kalibrację miał **słabszą**" — własna tabela pokazuje, że HGB ma **lepszy** Brier (0.2162/0.2156 vs 0.2174) i log-loss; „~3500 meczach treningowych" — faktycznie 1588 meczów (3176 próbek) | Teza odwrócona na zgodną z tabelą; liczba poprawiona |
| `TPM_Experiment_ValidateFeatures` | Cała tabela wniosków z poprzedniego runa (m.in. fatigue +0,03 zamiast −0,03, surface +0,60/p=0,105 zamiast +0,56/p=0,129, Elo +0,76/0,173 zamiast +0,73/0,195, baseline 0,6463 zamiast 0,6466) | Tabela przepisana z faktycznych outputów |
| `TPM_Experiment_WalkForward` | 4 nieaktualne liczby: baseline 0,6463→0,6466, +0,20→+0,17 p.p., p 0,66→0,72, „dodatnia w 3/6"→2/6 (output drukuje 2/6) | Poprawione |
| `TPM_Experiment_Elo` | „+0,8 p.p., p ≈ 0,17" vs output `delta=+0.0073, p=0.1945` | Poprawione na +0,73 / p ≈ 0,19 |
| `TPM_Experiment_Enriched` | Cytowane pooled +0,20/p=0,66 (stary run) vs +0,17/p=0,72; intro nazywało składniki „dwa wygrywające zestawy", choć na bieżącym teście oba są neutralne/ujemne | Poprawione |
| `TPM_Experiment_SurfaceSpeed` | Cytowane +0,60/p=0,11 vs +0,56/p=0,13 | Poprawione |
| `TPM_Experiment_MultiSeason` | „XGBoost z przodu i na trafności, **i na kalibracji**" — wg ECE najlepszy jest RF (0.0136 vs 0.0189); ponadto z notebooka ręcznie usunięto (po wykonaniu) dwa zastrzeżenia o nieporównywalności wyników multi-season vs single-season, które nadal są w `_gen_multiseason.py` | Teza doprecyzowana (XGB lepszy na Brier/log-loss, RF na ECE); oba zastrzeżenia przywrócone |

### 1.3. Notebooki nigdy nieuruchomione (ModelSlice + trio SliceAware) — naprawione

`TPM_Experiment_ModelSlice`, `TPM_Experiment_SliceAware`, `TPM_Experiment_SliceAware_BestOf5_v1`, `TPM_Experiment_SliceAware_QFServe_v3` mają **zero outputów i `execution_count: null`** — nigdy nie były wykonane w obecnej formie, a mimo to ich wnioski prezentowały liczby jako „Realne dane z uruchomienia". Trzy klasy problemów, wszystkie naprawione:

1. **Niedziałające ścieżki** (BestOf5, QFServe): `../data/sample_data/{rok}.csv` i `2024.csv` — pliki o takich nazwach już nie istnieją (po migracji na `atp_matches_{rok}.csv`), a `BASE_SCRIPT = Path("tennis_model.py")` było błędne z katalogu `notebooks/`. Naprawione: wzorzec `atp_matches_*`, lata sterowane env (`TENNIS_TARGET_YEAR`/`TENNIS_HISTORY_START` — spójnie ze `src/`), `../src/tennis_model.py`.
2. **Przestarzała, jednostronna metryka match-level** (tylko perspektywa `y==1`) — dokładnie ten błąd, który naprawiono w `src/` w Sprincie 1 (docstring `compute_symmetric_match_evaluation` wprost nazywa starą wersję zawyżającą). Notebooki dryfowały od źródeł. Naprawione: wszystkie trzy + opis w markdown używają teraz `compute_symmetric_match_evaluation` z namespace baseline'u.
3. **Wnioski oparte na unieważnionych liczbach**: „BestOf5 v1 +2,37 p.p. NAJLEPSZY" itd. pochodzą ze starego runa na 2024 ze złą metryką; po naprawie metryki ranking się odwrócił (SliceAware +1,69, BestOf5 +1,02, QFServe +0,17 — `docs/WYNIKI_SPRINTOW.md`), a walk-forward 2020–2025 nie potwierdził istotności żadnego wariantu. Naprawione: liczby oznaczone jako historyczne + dodana adnotacja z aktualnymi wynikami i odnośnikami. W ModelSlice poprawiono też wewnętrznie sprzeczne „61.02% (366 z 601)" → „(360 z 590)" (366/601 = 60,9%; 360/590 = 61,02% — wartość z raportu).

---

## 2. Uwagi do architektury modelu

**Architektura jest adekwatna do problemu i poprawnie złożona** — nie wymagała zmian:

- **Typ zadania**: predykcja prawdopodobieństwa zwycięzcy jako klasyfikacja binarna na danych symetryzowanych — poprawny wybór. Wyjście `predict_proba`, ewaluacja probabilistyczna (Brier, log-loss, ECE, reliability diagram) — spójna całość.
- **Symetryzacja** (2 wiersze na mecz, zamiana ról + negacja cech różnicowych i H2H) poprawnie eliminuje positional bias; etykiety idealnie zbalansowane 50/50. Zweryfikowano pole-po-polu, że każda cecha (w tym 33–60 nowych cech wariantów slice-aware) jest właściwie zamieniana w lustrzanym wierszu, a diffy liczone PO zamianie.
- **Dobór hiperparametrów po `neg_log_loss`** (nie accuracy) z multi-metric raportowaniem — dobra praktyka dla zadania probabilistycznego.
- **Kalibracja**: Platt scaling (`CalibratedClassifierCV` + `FrozenEstimator`) uczony na walidacji, nigdy na teście. Poprawne.
- **Stały próg 0.5 zamiast threshold tuningu** — świadoma, dobrze udokumentowana w kodzie decyzja (w danych symetrycznych tuning progu to gaming ewaluacji); historia tego błędu opisana w docstringu.
- **Random Forest jako model główny**: uzasadniony empirycznie — projekt porównał RF z HGB i XGBoost zarówno na małym (1 sezon), jak i dużym zbiorze (123 tys. próbek) i różnice mieszczą się w szumie.

Drobne ryzyka architektoniczne (świadomie pozostawione, do rozważenia):
- **TimeSeriesSplit na przeplatanych parach symetrycznych**: na granicy folda jedna para mecz/lustro może rozdzielić się między train i val (≤ n_splits meczów na tysiące — wpływ pomijalny). Czysty fix: split po parzystych indeksach lub po `match_id`.
- **`pressure_serve_score` w `bestof5_v1` jest dokładnym duplikatem `best_of5_serve_score`** (identyczne wejścia i fallback — w qfserve_v3 analogiczna funkcja ma odrębną gałąź late-rounds, tu wygląda na niedokończony port). 3 redundantne cechy rozcieńczają feature importance. Nie zmieniono (nie jest to błąd wyniku), zalecane usunięcie lub zróżnicowanie.
- **Globalny stan `_HISTORY_INDEX`/`_HISTORY_CUTOFF`** w qfserve_v3/bestof5_v1: `get_player_history` ignoruje argument `history`, gdy index jest ustawiony, a czyszczenie kontekstu nie jest w `try/finally` — wyjątek w pętli zostawia stan globalny. Dziś poprawne, ale to wzorzec, który już raz ugryzł (komentarz przy `tourney_path_*`).
- **`rest_days`/`tourney_minutes` (fatigue)**: `tourney_date` w danych Sackmanna to data STARTU turnieju, więc odpoczynek wewnątrz turnieju zawsze = 0, a po długim turnieju jest zawyżony o ~2 tygodnie. To ograniczenie danych, nie kodu — częściowo tłumaczy słabą ważność tych cech.
- **Monkey-patch `runpy.run_path` + współdzielony mutowalny namespace** w slicecompare/validate_variants: dziś bezpieczne (wszystkie warianty robią `.copy()`), ale jeden zapomniany `.copy()` w przyszłym wariancie po cichu skazi wyniki kolejnych modeli w tym samym runie.

---

## 3. Rozjazdy dokumentacja/notebooki ↔ kod

### Naprawione w tym przeglądzie (błędne definicje, nie tylko stale liczby)

- **`docs/SLOWNICZEK_POJEC.md`**: definicja Match Accuracy opisywała starą, jednostronną metrykę (sprzeczność z kodem i z WYNIKI_SPRINTOW, które nazywają ją bugiem) → przepisana na symetryczną; `statistically_below_overall` definiowane przez DOLNĄ granicę CI (bez sensu metodologicznie i niezgodne z kodem `ci_upper < overall_accuracy`) → poprawione; wybór hiperparametrów „po najwyższym wyniku [accuracy]" → doprecyzowane `refit='neg_log_loss'`; threshold tuning opisany jako przeszukiwanie siatki → opisany stan faktyczny (stały 0.5); `FrozenEstimator` „sklearn ≥1.8" → ≥1.6.
- **`docs/opis_tennis_model.md`**: `search` „wybiera po CV Accuracy" → neg_log_loss; `winner_perspective` „tylko jedna perspektywa" → uśrednienie obu; Q&A z błędnym mechanizmem match accuracy („obie perspektywy muszą się zgadzać") → opis uśredniania; FrozenEstimator 1.8 → 1.6.
- **`docs/opis_tennis_model_modelslice.md`**: wiersz tabeli `statistically_below_overall` podwójnie błędny („DOLNY brzeg > ogólne" = definicja bycia istotnie LEPSZYM); „Wald dla 0/4 daje [-0.1, 0.1]" — faktycznie daje [0,0], co poprawnie mówi Q&A w tym samym pliku → oba poprawione.

### Pozostawione (stale liczby / dokumenty historyczne — wymagałyby regeneracji, nie poprawek punktowych)

1. **Trzy pokolenia wyników i trzy różne „zwycięskie warianty"** w docs: `ai_model_slice.md`, `opis_tennis_model_sliceaware*.md`, `opis_tennis_model_slicecompare.md` i komentarz w `STRUKTURA_FOLDEROW.md` ogłaszają bestof5_v1 (+2,37, stara metryka); oba `reports/RAPORT_*warianty*` — qfserve_v3 (0,6271, inny run); `WYNIKI_SPRINTOW.md`/`PODSUMOWANIE_KONCOWE.md` (jedyne zgodne z bieżącym kodem) — sliceaware po naprawie metryki, a w walk-forward nic istotnego. Czytelnik bez kontekstu nie wie, która wersja obowiązuje.
2. **Nieaktualne liczby konfiguracyjne** rozsiane po starszych docs: rok 2024 + historia 2018–2023 (obecnie 2025 + 2001–2024), 590 meczów testu, headline 61,02% (bieżąca konfiguracja daje ~65,7%), liczba cech wariantów (sliceaware 33 nowe/73 razem, nie „32/72"; bestof5 39/79, nie „37/77"; qfserve_v3 60/100, nie „~50/90"), siatki hiperparametrów w SLOWNICZKU bez wartości 500/None/1.
3. **`STRUKTURA_FOLDEROW.md`** wymienia 7 plików w `src/` (jest 18), stare nazwy danych, nieistniejące `README.md`/`requirements.txt`.
4. **`RAPORT_tennis_model_sliceaware_i_slicecompare.md`** §13 twierdzi, że slicecompare porównuje 3 modele bez sliceaware (kod ma 4, z sliceaware) i że wynik zapisuje się do CSV (kod zapisuje XLSX).
5. **Cross-doc**: ten sam slice R128×L-vs-R raz „+44,4 p.p.", raz „+33,3 p.p."; przykład arytmetyczny w SLOWNICZKU (64%×0.82+40%×0.18=„56%") nie składa się przy żadnej interpretacji; cytat „Godfrey et al., VLDB 2025" w ModelSlice — niespotykany nigdzie indziej, prawdopodobnie konfabulacja do weryfikacji.
6. **Niedeterminizm środowiska**: identyczny pipeline (seed 42) daje 0,6566 lokalnie vs 0,6585 w Colab (różnica 1 meczu — prawdopodobnie remis dokładnie 0,5 przy ostrym `>`). Część rozjazdów liczb między notebookami to właśnie to.
7. **Archiwa** (`experiments_archive/*.py`, `notebooks/archive/*`): stara jednostronna metryka, tuning okna na zbiorze testowym (Window_Size), wybór klasyfikatora na teście z pominięciem walidacji (Classifiers), nieuruchomione Stability porównujące różne zbiory testowe jako identyczne. Pozostawione bez zmian jako snapshoty historyczne — ale `tennis_model_ewma.py` porównuje swój stary wynik z hardkodowanym 0,6153 nowej metryki (mieszanie metryk) i deklaruje „48 cech" main.py (jest 40).

### Dokumenty w pełni zgodne z kodem

`docs/PODSUMOWANIE_KONCOWE.md`, `docs/WYNIKI_SPRINTOW.md`, `docs/opis_tennis_model_multiseason.md`, `docs/OPIS_DLA_PROMOTORA.md` (z drobnym zastrzeżeniem co do „warianty okazały się słabsze" — bestof5 był +0,56 p.p. n.s., kierunkowo dodatni). Najzdrowszy notebook: `TPM_Experiment_Salvage` (sekwencyjny, generator odtwarza go bajt-w-bajt, wnioski cytują outputy wprost).

---

## 4. Mocne strony projektu

1. **Dyscyplina anty-leakage na poziomie produkcyjnym**: expanding window z twardym `cutoff` (bisect po posortowanych indeksach), historia walidacji = historia + train, historia testu = historia + train + val; enkodery w multiseason fitowane wyłącznie na treningu; `court_pace_index` wyłącznie z lat < target; Elo z poprawnym porządkiem „cechy przed update'em" i K-factorem wg FiveThirtyEight.
2. **Symetryczna ewaluacja match-level** (uśrednienie obu perspektyw) — rozwiązanie nieoczywistego problemu, z docstringiem dokumentującym, dlaczego stara wersja była błędna i dlaczego threshold tuning na symetrycznych danych nie ma sensu. Rzadko spotykana samoświadomość metodologiczna.
3. **Kultura walidacji statystycznej**: parowany McNemar po `match_id`, walk-forward po 6 sezonach jako odpowiedź na szum pojedynczego test setu, Wilson CI dla małych slice'ów, seed-stability. Projekt sam wykrył, że swoje „+2,37 p.p." było artefaktem metryki — i to udokumentował.
4. **Świadoma praca z probabilistyką**: refit po log-loss, Brier/ECE/reliability diagram, kalibracja Platta na walidacji.
5. **Inżynieria**: wektoryzacja serve stats, indeks gracza O(log K) zamiast skanów O(N), cache namespace'u baseline przy porównaniach wielomodelowych, asserty wyrównania ramek kontekstowych, `validate="one_to_one"` w merge'ach, walidacja formatu `tourney_id` chroniąca przed leakage w `tourney_path_*`.
6. **Dokumentacja wyjaśniająca „dlaczego", nie tylko „co"** (sekcje Q&A dla promotora, słowniczek) — mimo że część zdążyła się zestarzeć, sam standard jest wysoki.

---

## 5. Rekomendacje na przyszłość (priorytetyzowane)

**P1 — spójność wyników (największy zwrot przy małym koszcie):**
1. Przeprowadzić jednorazową „konsolidację prawdy": w starych docs (`ai_model_slice.md`, `opis_*sliceaware*`, `opis_*slicecompare*`, oba `RAPORT_*warianty*`, komentarz w `STRUKTURA_FOLDEROW.md`) dodać na górze banner „wyniki historyczne, stara metryka — aktualne: WYNIKI_SPRINTOW.md" albo zaktualizować liczby. Obecnie trzy dokumenty ogłaszają trzy różnych zwycięzców.
2. Przestać hardkodować wnioski w `_gen_*.py` — generować tabelę wniosków z faktycznych wyników runa (albo przynajmniej wstawiać placeholder „uzupełnić po uruchomieniu"). To źródło 7 z 8 rozjazdów w notebookach.
3. ✅ **Wykonane (2026-06-11, po przeglądzie)**: 4 notebooki slice-aware/ModelSlice uruchomione na bieżącej konfiguracji — szczegóły w sekcji „Aktualizacja" na końcu raportu.

**P2 — drobne usprawnienia metodologiczne:**
4. Wyjaśnić niedeterminizm 0,6566 vs 0,6585 (lokalnie vs Colab; podejrzenie: mecz z probą dokładnie 0,5 przy ostrym `>` — można np. raportować remisy osobno) i ujednolicić środowisko uruchomień.
5. Kontrola wielokrotnych porównań (np. Holm) w `salvage`/`validate_features` — 4 warianty × McNemar bez korekty.
6. Split CV po `match_id`/parzystych indeksach, żeby para symetryczna nigdy nie rozdzielała się na granicy folda.
7. Usunąć lub zróżnicować `pressure_serve_score` w `bestof5_v1` (duplikat `best_of5_serve_score`); usunąć martwą zmienną `level_strength` (bestof5_v1:525) i nieużywany `THRESHOLD_GRID` (tennis_model.py).
8. Owinąć `set_history_context` w `try/finally` (qfserve_v3, bestof5_v1) i rozważyć przekazywanie indeksu jawnie zamiast stanu modułowego.

**P3 — opcjonalne kierunki rozwojowe:**
9. Parametr `window` w `calculate_context_experience` jest no-opem (zawsze `window > scale`) — usunąć lub zamienić na okno czasowe.
10. Przy porównaniach slicecompare rozważyć dodanie wariantu na probach skalibrowanych (dziś wszystkie modele porównywane na raw — wewnętrznie spójne, ale model „wdrażalny" jest kalibrowany).
11. Dane: rozważyć zewnętrzne źródło dat meczów (nie startu turnieju), co odblokowałoby sensowne `rest_days`; obecna wersja jest systematycznie zgrubna.
12. `experiments_archive/` — dopisać do nagłówków plików jedno zdanie „stara jednostronna metryka, wyniki nieporównywalne z bieżącym pipeline" (świadomie nie zmieniałem kodu archiwów).

---

## Aneks: pełna lista zmodyfikowanych plików (26)

**src/ (11):** `tennis_model.py` (docstring), `tennis_model_elo.py` (McNemar, context_cols), `tennis_model_walkforward.py` (McNemar, docstring), `tennis_model_salvage.py` (McNemar, mianownik), `tennis_model_validate_features.py` (McNemar, martwe wyrażenie), `tennis_model_validate_variants.py` (McNemar), `tennis_model_sliceaware.py` (is_lefty_matchup), `tennis_model_sliceaware_qfserve_v3.py` (is_lefty_matchup), `tennis_model_surface_speed.py` (NaN-guard, docstring), `tennis_model_ewma_ablation.py` (kolejność update'ów EWMA), `tennis_model_seedstability.py` (fallback cv_accuracy).

**notebooks/ (12):** Fatigue, HGB, Elo, WalkForward, Enriched, SurfaceSpeed, ValidateFeatures, MultiSeason (korekty wniosków/przywrócone zastrzeżenia) oraz ModelSlice, SliceAware, SliceAware_BestOf5_v1, SliceAware_QFServe_v3 (ścieżki danych, symetryczna metryka, oznaczenie historycznych liczb).

**docs/ (3):** `SLOWNICZEK_POJEC.md`, `opis_tennis_model.md`, `opis_tennis_model_modelslice.md` (błędne definicje).

Żaden plik w `experiments_archive/`, `notebooks/archive/` ani `reports/` nie był modyfikowany (snapshoty historyczne — uwagi w sekcji 3). Zgodnie z ustaleniem nie tworzono commitów.

---

## Aktualizacja (2026-06-11, po przeglądzie): uruchomienia weryfikacyjne

Po naprawach wykonano 4 wcześniej nieuruchamiane notebooki (sekwencyjnie, nbclient, sezon 2025, metryka symetryczna, RANDOM_STATE=42) i zaktualizowano ich wnioski o realne liczby:

| Notebook | Wynik | Czas |
|---|---|---|
| ModelSlice | baseline match acc **65.66%** (348/530); 349 slice'ów, tylko **5** istotnie poniżej średniej; Bo5 NIE jest już słabym slicem (słabe: R128 59.4%, Masters×rank_gap>100 36.8% sig.) | 91 s |
| SliceAware | 65.28%, delta **−0.38 p.p.** (val +1.42 / test −1.51 — brak generalizacji) | 208 s |
| SliceAware_BestOf5_v1 | 66.60%, delta **+0.94 p.p.** — najlepszy wariant; `endurance_score_diff` i `best_of5_return_score_diff` w czołówce ważności | 196 s |
| SliceAware_QFServe_v3 | 63.58%, delta **−2.08 p.p.** — najgorszy; val +1.61 / test −2.17 = overfitting przy 100 cechach | 283 s |

Nowy ranking wariantów na 2025 (BestOf5 > baseline > SliceAware > QFServe) jest spójny kierunkowo z walk-forward 2020–2025 (+0.56 / −0.26 / −0.89 p.p., wszystkie n.s.) i ostatecznie zamyka historię trzech sprzecznych rankingów.

Uruchomiono też `tennis_model_elo.py` (walk-forward 6 sezonów): pooled 3022 mecze, baseline 64.63% → +Elo 65.39%, delta **+0.76 p.p.**, dodatnia w 4/6 sezonów, McNemar b=119 c=142 z=1.36 **p=0.17** → brak istotności. Cechy Elo mają najwyższe ważności w modelu (elo_diff średni rank 2.2/44), ale duplikują sygnał rankingu ATP + formy.

### Aktualizacja 2 (2026-06-12): rerun WSZYSTKICH głównych notebooków

Na życzenie wykonano ponownie pozostałe 11 głównych notebooków (lokalnie, nbclient; `archive/` celowo pominięte jako zamrożone snapshoty). **Efekt: pełna spójność liczb w całym projekcie** — każdy notebook raportuje teraz baseline 2025 = 0.6566 i pooled 2020–2025 = 0.6463; zniknął rozjazd 0.6566/0.6585 (lokalnie vs Colab — lokalny rerun odtworzył wartości pierwotnych runów, czyli to re-wykonania w Colab były „przesunięte" o 1–2 mecze).

- **Wykonane przez nbclient (10):** EWMA (65 s), HGB (94 s), Fatigue (61 s), SurfaceSpeed (56 s), Enriched (55 s), Elo (281 s), WalkForward (279 s), Salvage (280 s), ValidateFeatures (320 s), MultiSeason (5736 s). Wnioski 7 z nich zsynchronizowano z nowymi outputami (zmiany rzędu 1–2 meczów: m.in. VF — surface +0.60/p=0.105, fatigue +0.03/p=1.0, enriched +0.20/p=0.656, elo +0.76/p=0.173; WalkForward +0.20/p=0.66, 3/6 sezonów); HGB, MultiSeason i Salvage nie wymagały zmian. Wynik EWMA (−0.57 p.p.) odzwierciedla już naprawę seedowania surface-EWMA z przeglądu.
- **ValidateVariants — wykonany przez src, zweryfikowany 1:1:** notebook dwukrotnie przekroczył timeout komórki w nbclient (7200 s i 21600 s) mimo że identyczne obliczenia w czystym procesie Pythona zajmują <1 h — przyczyną jest zawieszanie się komunikacji kernela Jupyter (zmq na Windows; w logu błąd `IPKernelApp … zmq send`) przy długiej pętli spawnującej dziesiątki procesów loky, a nie kod projektu. Zamiast tego uruchomiono `src/tennis_model_validate_variants.py` bezpośrednio: **wszystkie 18 wartości per-sezon i 3 statystyki pooled (sliceaware −0.26 p.p./p=0.610; bestof5_v1 +0.56/p=0.307; qfserve_v3 −0.89/p=0.076) są identyczne co do meczu z istniejącymi outputami notebooka** — outputy i wnioski ValidateVariants są więc aktualne bez ponownego zapisu.
- **Znany pozostały dryf:** kopia BestOf5 w notebooku ma 33 dodane cechy, a `src/tennis_model_sliceaware_bestof5_v1.py` — 39 (w notebooku brak rodzin `best_of5_surface_form` i `best_of5_vs_top30_form`); stąd 2025: notebook 0.6660 (+0.94 p.p.) vs src w ValidateVariants 0.6679 (+1.13 p.p.). Oba artefakty są wewnętrznie spójne; docelowo warto ujednolicić kopię w notebooku ze src.
- Uruchomiono też pełny `tennis_model.py` (wyniki jak w sekcji wyżej: match 65.66%, Brier 0.2172/ECE 0.0367 po kalibracji); przy okazji naprawiono zahardkodowany rok w printcie nagłówka podziału. Uwaga środowiskowa: w venv brakuje matplotlib, więc reliability diagram jest pomijany.
