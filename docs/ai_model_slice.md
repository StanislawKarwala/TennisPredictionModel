# Model Slicing w modelu predykcji meczów tenisowych

Dokument tłumaczy prostym językiem, o co chodzi w czterech plikach (`tennis_model_modelslice.py`, `tennis_model_sliceaware.py`, `tennis_model_sliceaware_bestof5_v1.py`, `tennis_model_sliceaware_qfserve_v3.py`) i odpowiadających im notebookach `TPM_Experiment_*.ipynb`. Jest pomyślany jako ściąga przed rozmową z promotorem.

---

## 1. Punkt wyjścia: o co chodziło z `tennis_model.py`

Model bazowy to **Random Forest** trenowany na meczach z 2024 roku. Predykcja: który z dwóch graczy wygra. Po pełnym pipeline (chronologiczny split 60/20/20, expanding window dla cech dynamicznych, symetryzacja, RandomizedSearchCV) model osiąga:

| Metryka | Wartość |
|---|---|
| CV Accuracy (TimeSeriesSplit) | **64.78%** |
| Validation Accuracy | **63.31%** |
| Test Accuracy (symetryzowana) | **61.36%** |
| **Match Accuracy (poziom meczu)** | **61.02%** |
| Brier Score (test) | 0.2284 |
| Log-loss (test) | 0.6464 |
| ECE (Expected Calibration Error) | 0.0403 |

Match Accuracy 61.02% oznacza: na 601 meczów testowych model poprawnie typuje 366. Przewaga nad rzutem monetą: +11 p.p.

Problem: **64% to średnia po wszystkich meczach**. Nie wiemy, czy model:
- jest niezły wszędzie i tu i tam się myli,
- czy zawala konkretne typy meczów (np. zawsze przegrywa na Bo5, ale świetnie radzi sobie z Hard Courtem).

Bez tej wiedzy nie wiemy, **gdzie warto coś poprawiać**.

---

## 2. `tennis_model_modelslice.py` — diagnostyka „gdzie konkretnie model się myli"

**Pomysł** pochodzi z artykułu *Model Slicing for Responsible AI* (Godfrey et al., VLDB 2025, plik `GuideAI25_2.pdf`). Skrót *slice* = „kawałek" — bierzemy testowe mecze i kroimy je na semantyczne podgrupy, np.:
- wszystkie mecze na trawie,
- wszystkie ćwierćfinały,
- wszystkie mecze leworęczny vs praworęczny,
- wszystkie mecze gdzie różnica rankingowa jest >100 pozycji.

Dla każdej takiej podgrupy liczymy accuracy. Jeśli ogólne accuracy to 64%, a w slice'ie „Best of 5" wynosi 52%, to mamy konkretny target do poprawy.

### Co robi ten plik
1. Uruchamia bazowy pipeline (`tennis_model.py`) i pobiera wyniki testowe.
2. Każdemu meczowi przypisuje atrybuty slicingowe (surface, round, best_of, handedness_matchup, rank_gap_bucket, age_gap_bucket, form_gap_bucket, tourney_level).
3. Generuje **wszystkie kombinacje 1D i 2D atrybutów** (np. `surface=Grass`, albo `surface=Grass & round=QF`).
4. Dla każdej grupy z support ≥ 5 liczy:
   - **accuracy** i **error_rate**,
   - **Wilson Confidence Interval (95%)** — bo proste „4 z 5 = 80%" jest niewiarygodne przy małym n; Wilson daje sensowne granice (np. \[37.6%, 96.4%\]),
   - flagę `statistically_below_overall = True` gdy górny brzeg CI < accuracy ogólne (czyli z 95% pewnością slice jest gorszy, a nie tylko pechowy szum),
   - **Brier Score** = mean((1 - p_zwycięzca)²) — kara za błędne pewne predykcje,
   - **Log Loss** — standardowa metryka kalibracji.
5. Drukuje najsłabsze slice'y 1D, 2D oraz najlepsze (dla kontrastu).

### Co z tego wynika (typowe obserwacje)
Najczęściej powtarzające się słabe podgrupy:
- **`best_of=5`** — Grand Slamy. Tylko ~18% danych, więc RF mało się z nimi mierzy.
- **`round=QF`** — ćwierćfinały, gdzie spotykają się wyrównani gracze; ranking traci moc.
- **`handedness_matchup=L-vs-R`** — leworęczny przeciw praworęcznemu.
- **`rank_gap=0-10`** — top vs top, decyduje forma dnia, nie pozycja w rankingu.

To są **konkretne targety do poprawy** — i o tym są kolejne pliki.

---

## 3. Trzy warianty „slice-aware" — model uczy się o słabych miejscach

Każdy z trzech plików stosuje tę samą strategię:
1. Wczytuje baseline (cały pipeline `tennis_model.py`) bez zmian.
2. Dla każdego meczu policza **dodatkowe cechy specyficzne dla słabych slice'ów**.
3. Trenuje **nowy Random Forest na rozszerzonych cechach**, ale z **tymi samymi hyperparams co baseline** (`baseline_search.best_params_`). To kluczowe — porównanie pokazuje wpływ samych cech, a nie tuningu.
4. Porównuje match accuracy z baseline.

### 3a. `tennis_model_sliceaware.py` — atak na trzy słabe slice'y naraz

Najprostszy z trójki. Dodaje **20 cech kontekstowych**:
- **Best of 5 form / experience** — winrate i doświadczenie gracza w meczach Bo5
- **Late round form / experience** — winrate w QF/SF/F (presja końcówek)
- **vs opp hand form** — winrate gracza przeciw konkretnej ręczności rywala (np. „Djokovic vs leworęczni")
- **vs opp hand surface form** — to samo, ale na bieżącej nawierzchni
- **QF form / experience / surface form** — bardziej granularne dla ćwierćfinałów
- **vs opp hand balance** — wins minus losses przeciw konkretnej ręczności

Wszystkie liczone z **expanding window** (tylko mecze rozegrane wcześniej niż bieżący). Fallback na ogólną formę gdy gracz nie ma minimum 2-3 meczów w danym kontekście.

**Realny efekt (uruchomienie z RANDOM_STATE=42)**: match accuracy **60.85%** vs baseline 61.02% — **delta -0.17 p.p.**, czyli praktycznie bez zmian. Dodanie 20 cech kontekstowych dla 3 słabych slice'ów naraz nie poprawia średniej, bo:
- na targetowych slice'ach faktycznie pomaga (np. `L-vs-R × tourney_level=500` +8.6 p.p., `L-vs-R × rank_gap=>100` +7.7 p.p.)
- ale w innych miejscach szkodzi (np. `tourney_level=F` -20 p.p., `round=QF × form_gap=0.00-0.10` -10.5 p.p.)
- te dwa efekty się równoważą na poziomie ogólnym

Wniosek: dodanie szerokiego zestawu cech *bez głębokiej walidacji per slice* potrafi przesunąć błędy zamiast je usunąć.

### 3b. `tennis_model_sliceaware_bestof5_v1.py` — głębokie wejście w Bo5

Zamiast atakować 3 słabe slice'y po trochu, ten wariant **idzie głębiej tylko w Best of 5**. Dodaje cechy specyficzne dla dystansu pięciosetowego:
- **Long match form** — winrate w meczach trwających >150 minut
- **Best of 5 avg minutes** — średni czas trwania Bo5 gracza
- **Best of 5 serve / return / stability** — jakość serwisu i powtarzalność na Bo5
- **Pressure serve score** — serwis pod presją (Bo5)
- **Best of 5 vs top30 form** — czy gracz radzi sobie w Bo5 przeciw faworytom
- **Endurance score** — kompozytowy wskaźnik wytrzymałości (waga: 25% Bo5 form + 20% long-match form + 15% Bo5 exp + 10% long exp + 15% normalized minutes + 15% serve stability)
- **Tournament level strength** — numeryczne `G > M > F > 500 > 250`

Wymaga doczytania kolumny `minutes` z 2024.csv (nie ma jej w bazowym cols_base).

**Realny efekt (RANDOM_STATE=42)**: match accuracy **63.39%** vs baseline 61.02% — **delta +2.37 p.p.** Najlepszy wynik spośród wszystkich wariantów. Cechy `endurance_score`, `best_of5_serve_score`, `tourney_level_strength` wchodzą do top 30 ważności. Największe zyski na targetowych slice'ach:
- `L-vs-R × rank_gap=0-10` (top vs top z różną ręcznością): +21.4 p.p.
- `tourney_level=M × L-vs-R` (Masters z lewo): +17.6 p.p.
- `rank_gap=0-10 × age_gap=>8` (wyrównani rankingiem, duża różnica wieku): +33.3 p.p.

Słabe miejsca: `tourney_level=F` (finały) -20 p.p. — model nie ma wystarczająco danych z F (małe support).

### 3c. `tennis_model_sliceaware_qfserve_v3.py` — najbogatszy, kontekst turniejowy + warunkowy serwis

Najambitniejszy wariant. Łączy wszystkie cechy ze SliceAware z dwoma nowymi kierunkami:

**A. Kontekst turniejowy (QF v3)** — informacje o turnieju, których baseline w ogóle nie ma:
- **Seed context score** — pozycja rozstawienia gracza w drabince. Top seed → 1.0, ostatni seed → ~0.0. Soft proxy z rankingu gdy seed = NaN (większość graczy nie jest rozstawiona).
- **Tourney path opp strength** — średnia siła (log rank points) rywali, których gracz już pokonał w bieżącym turnieju. Mówi: „w QF Wimbledonu gracz A pokonał #50→#30→#20 — to lepsza droga niż gracz B który pokonał #200→#150→#100".
- **Tourney path match count** — ile meczów rozegrał już w tym turnieju.
- **QF level pressure** = `is_qf × tourney_level_strength` — interakcja, bo QF Grand Slamu ≠ QF ATP 250.

**B. Warunkowy serwis (Serve v2)** — serwis nie jest statyczny:
- **Surface serve score** — serwis na bieżącej nawierzchni
- **Top opponent serve score** — serwis przeciw top 20/30
- **vs opp hand return score** — return przeciw konkretnej ręczności rywala
- **Surface serve stability** — czy serwis jest powtarzalny na tej nawierzchni
- **Pressure serve score** — serwis w Bo5 lub w QF/SF/F

**Walidacja `tourney_id`**: dodany asercjonalny check formatu `YYYY-...`, bo `tourney_path_*` filtruje po `tourney_id` i bez prefiksu roku doszłoby do leakage'u (te same numery ID powracają z roku na rok).

**Realny efekt (RANDOM_STATE=42)**: match accuracy **63.22%** vs baseline 61.02% — **delta +2.20 p.p.** Drugi najlepszy wynik (minimalnie za BestOf5 v1 +2.37 p.p.). Spektakularne zyski na konkretnych slice'ach:
- `round=R128 × L-vs-R` (pierwsza runda Grand Slamu z lewo): baseline 33.3%, qfserve_v3 **77.8%** (+44.4 p.p.)
- `round=R128 × rank_gap=51-100`: +40.0 p.p.
- `round=R128 × rank_gap=11-25`: +33.3 p.p.
- `tourney_level=250 × QF`: baseline 44.4%, qfserve_v3 51.8% (+7.4 p.p. — faktycznie poprawia QF)
- `L-vs-R × form_gap=>0.40`: baseline 50%, qfserve_v3 66.7% (+16.7 p.p.)

Słabe miejsca: `round=R128 × rank_gap=26-50` -33.3 p.p., `tourney_level=F × age_gap=0-2` -33.3 p.p. — model traci dokładność w niektórych podgrupach.

Najdroższy obliczeniowo — ~25 wywołań filtrów per mecz.

---

## 4. Co z tego razem wynika

**Podsumowanie real numerów** (RANDOM_STATE=42, run z pełnego pipeline'u):

| Model | Match accuracy | Delta vs baseline |
|---|---|---|
| Baseline `tennis_model.py` | 61.02% | — |
| SliceAware (3 słabe slice'y) | 60.85% | -0.17 p.p. |
| **QFServe v3** (seed + serve) | **63.22%** | **+2.20 p.p.** |
| **BestOf5 v1** (endurance) | **63.39%** | **+2.37 p.p.** |

Najważniejsze obserwacje:

1. **Focused approach wygrywa**: warianty głęboko skupione na konkretnym kontekście (BestOf5, QFServe) dają **+2.2 do +2.4 p.p.**, podczas gdy shotgun po wszystkich słabych slice'ach (SliceAware) jest **netto bez zmian**.
2. **Na targetowych slice'ach efekt jest spektakularny**: pierwsza runda Grand Slamu z meczem L-vs-R zyskuje **+44.4 p.p.** dzięki QFServe v3 (z 33.3% na 77.8%). To ekstremum, ale pokazuje że dobrze ukierunkowane cechy mogą gigantycznie poprawić problematyczne podgrupy.
3. **Trade-off jest realny**: warianty poprawiają targetowe slice'y, ale czasem szkodzą innym (np. finały, gdzie support jest mały). Suma efektów na poziomie ogólnym to +2 p.p., ale rozkład jest nierównomierny.

**Praktyczna konkluzja dla promotora**: model slicing to przede wszystkim **narzędzie diagnostyczne** — pokazuje konkretne podgrupy do poprawy. Same cechy slice-aware dają poprawę +2-2.5 p.p. na poziomie meczu, co jest wynikiem solidnym dla domeny tenisowej (literatura raportuje 65-70% jako sufit dla modeli featurowych w tenisie męskim — głównie ze względu na szum w danych: kontuzje, presja, forma dnia).

---

## 5. Plik `tennis_model_slicecompare.py` — porównanie wariantów

Uruchamia wszystkie 4 modele (baseline + 3 slice-aware), buduje **wspólną tabelę slice'ów** i pokazuje delty per slice per model. Wynik zapisuje do `slice_comparison_all_variants.xlsx` z arkuszami:
- `overall_metrics` — overall accuracy + delta vs baseline dla każdego modelu
- `slice_comparison` — pełna tabela z accuracy w każdym slice'ie i kolumny `*_accuracy_delta_vs_baseline`

Dodatkowo flaguje target slice'y (`best_of=5`, `round=QF`, `handedness_matchup=L-vs-R`) i sortuje wyniki tak, żeby te najważniejsze były na górze.

Optymalizacja: cache'uje namespace baseline pipeline (50-iteracyjny RandomizedSearchCV uruchamia się **raz**, a nie 4 razy).

---

## 6. Co pokazują notebooki `TPM_Experiment_*.ipynb`

Cztery notebooki Jupyter są wersją demonstracyjną tych samych eksperymentów. Każdy podzielony jest na małe komórki z opisami:

- **`TPM_Experiment_ModelSlice.ipynb`** — Wilson CI, jak działa slicing, wnioski o słabych podgrupach. Zaczyna od pokazania, czemu naiwne CI nie wystarcza, tłumaczy log-loss/brier per slice, kończy listą najsłabszych slice'ów.
- **`TPM_Experiment_SliceAware.ipynb`** — pierwszy slice-aware wariant. Pokazuje player history index z bisect, funkcje kontekstowe (form/experience/balance), symetryzację p1/p2, porównanie z baseline.
- **`TPM_Experiment_SliceAware_BestOf5_v1.ipynb`** — wariant Bo5. Pokazuje endurance_score (z wagami), serve score (z 8 metryk → 1 skalar), ładowanie `minutes` z 2024.csv.
- **`TPM_Experiment_SliceAware_QFServe_v3.ipynb`** — najbogatszy wariant. Pokazuje seed context score (z fallbackiem), tournament path stats, walidację tourney_id, warunkowy serwis.

Każdy notebook kończy się sekcją „Wnioski" — co dało nowe cechy, jakie są ograniczenia, jakie kierunki dalej.

---

## 7. Słowniczek dla promotora (krótko)

| Pojęcie | Tłumaczenie |
|---|---|
| **Slice** | Podgrupa testowych meczów wybrana po atrybutach (np. mecze QF na trawie) |
| **Support** | Liczba meczów w slice'ie — mały support = niewiarygodne accuracy |
| **Wilson CI** | Sensowny 95% przedział ufności dla proporcji przy małym n; lepszy niż naiwne ±z·√(p(1-p)/n) |
| **Symetryzacja** | Każdy mecz daje 2 wiersze (p1=winner / p1=loser) — eliminuje positional bias |
| **Expanding window** | Cechy dynamiczne (forma) liczone tylko z meczów *rozegranych wcześniej* — brak data leakage |
| **Best params reuse** | Slice-aware używa hyperparams baseline'u, żeby porównanie było uczciwe (różnica = cechy, nie tuning) |
| **PlayerHistoryIndex** | Mapa „gracz → indeksy w historii" z `bisect_left` — przyspiesza feature engineering 3-8× |
| **Brier Score** | Średni błąd kwadratowy prawdopodobieństw; im niżej, tym lepiej; 0.25 = losowe |
| **Log Loss** | -mean(log(p_prawdziwy_zwyciezca)); standardowa metryka kalibracji; ~0.69 = losowe |
| **Endurance score** | Kompozytowy wskaźnik wytrzymałości w Bo5 (suma ważona 6 składników) |
| **Seed context score** | Numeryczny wskaźnik rozstawienia w turnieju, z fallbackiem na ranking |
| **Tourney path opp strength** | Średnia siła rywali pokonanych w bieżącym turnieju (mówi: trudna czy łatwa droga) |
