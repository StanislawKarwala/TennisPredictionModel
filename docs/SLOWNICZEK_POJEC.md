# Słowniczek pojęć — Co znaczy każde słowo z raportów

**Po co ten plik:** zebrałem tu wszystkie techniczne pojęcia z plików `opis_*.md` i raportów, wytłumaczone najprostszymi słowami. Możesz zerknąć tutaj, gdy promotor zapyta „a co to właściwie znaczy…".

---

## Podstawy modelu

### Random Forest (Las losowy)
Algorytm, którego używamy do predykcji. To „komitet drzew decyzyjnych":
- Buduje 100-300 osobnych drzew decyzyjnych
- Każde drzewo trenuje na nieco innym fragmencie danych (bootstrap sampling) i na nieco innym podzbiorze cech (max_features)
- Predykcja końcowa = średnia/głosowanie wszystkich drzew

**Czemu nie pojedyncze drzewo?** Pojedyncze drzewo zapamiętuje dane treningowe (overfitting). Las uśrednia szum z różnych drzew → bardziej stabilna predykcja.

**Czemu Random Forest, nie sieć neuronowa?** Dla ~3000 meczów rocznie sieć neuronowa się nie nauczy (potrzebuje milionów przykładów). RF dobrze pracuje na małych ustrukturyzowanych danych, sam wybiera ważne cechy, nie wymaga normalizacji.

### Klasyfikator binarny
Model rozróżnia dwie klasy: **y=1** (Gracz 1 wygrał) lub **y=0** (Gracz 1 przegrał). Zwraca prawdopodobieństwo P(y=1) — np. „0.73 szansy że Gracz 1 wygra".

### Cecha (feature)
Liczba opisująca mecz, którą model widzi na wejściu. Np. `p1_rank_log` (logarytm rankingu Gracza 1), `p1_form` (jego forma z ostatnich 10 meczów). Mamy 40 cech w baseline, 60-90 w wariantach.

### Hiperparametr
Ustawienie modelu, którego sam się nie nauczy — trzeba je wybrać przed treningiem. Dla Random Forest np.: ile drzew (`n_estimators=100/200/300`), maksymalna głębokość drzewa (`max_depth=10/20/30`), minimalna liczba próbek w liściu (`min_samples_leaf=2/5/8`). Hiperparametry kontrolują kompromis między dopasowaniem a uogólnieniem.

### RandomizedSearchCV
Sposób na automatyczne znalezienie najlepszego zestawu hiperparametrów. Zamiast testować wszystkie kombinacje (grid search → bardzo wolne), losowo wybiera 50 zestawów hiperparametrów, ocenia każdy przez cross-validation i wybiera ten, który dał najwyższy wynik. „Random" = losowy wybór kombinacji do testowania.

---

## Cztery rodzaje „Accuracy"

To najczęstsze pojęcia w raportach — każde co innego znaczy.

### CV Accuracy (Cross-Validation Accuracy) — u nas ~64.78%
**Co to:** średnia trafność modelu mierzona na zbiorze TRENINGOWYM podzielonym na 5 części.
**Jak liczymy:**
1. Dzielimy trening na 5 fold-ów (po ~700 meczów)
2. Trening na fold-ach 1-4, test na fold-zie 5 → accuracy A
3. Trening na fold-ach 1-3+5, test na fold-zie 4 → accuracy B
4. ... powtarzamy 5 razy z różną kombinacją
5. Liczymy średnią z A, B, C, D, E

**Po co to:** żeby porównać różne zestawy hiperparametrów BEZ zaglądania w zbiór walidacyjny i testowy. CV pozwala uczciwie ocenić „czy ten zestaw hiperparametrów jest lepszy od tamtego".

**Dlaczego TimeSeriesSplit a nie zwykły KFold?** Tenis to dane czasowe — model nie może widzieć meczu z czerwca podczas treningu, a potem przewidywać meczu z marca (to byłby leakage z przyszłości). TimeSeriesSplit dba o to, że trening jest ZAWSZE chronologicznie wcześniejszy niż test w danym fold-zie.

### Validation Accuracy — u nas ~63.31%
**Co to:** trafność na zbiorze WALIDACYJNYM (20% danych, oddzielonym wcześniej, niewidzianym przez model podczas treningu).
**Po co to:** sanity check ostatecznego modelu (z wybranymi już hiperparametrami) zanim spojrzymy na test set. Jeśli validation acc jest dużo gorsza od CV acc, model się przeucza.

### Test Accuracy — u nas ~62.46%
**Co to:** trafność na zbiorze TESTOWYM (ostatnie 20% danych, finalna ocena).
**Po co to:** to jest „uczciwa" liczba do raportowania światu. Test set ma być zerową ingerencją — patrzymy raz, na końcu.

**Czemu zwykle Test Accuracy < Validation Accuracy < CV Accuracy?** Im świeższe dane (test = najnowsze), tym większa szansa zaskakujących wzorców (kontuzje, transfer formy, nowi gracze) których model nie widział w treningu. Spadek o 1-2 p.p. między CV a test jest normalny.

### Match Accuracy — u nas 61.02%
**Co to:** trafność na poziomie MECZU, nie wiersza. Po symetryzacji każdy mecz daje 2 wiersze: jeden gdzie Gracz 1 = winner (y=1), drugi gdzie Gracz 1 = loser (y=0). Test Accuracy liczy oba.
**Match Accuracy = inny widok:**
1. Bierzemy tylko jedną perspektywę meczu (np. „prawdziwy winner jako Gracz 1")
2. Model przewiduje P(y=1) — szansa że Gracz 1 wygra
3. Jeśli P(y=1) > 0.5 → predykcja „Gracz 1 wygra" → POPRAWNA (bo Gracz 1 to faktyczny winner)
4. Jeśli P(y=1) ≤ 0.5 → predykcja „Gracz 2 wygra" → BŁĘDNA
5. Match Accuracy = % poprawnie przewidzianych meczów (z perspektywy winner)

**Po co osobno Match Accuracy?** Bo to jest najprostsza, najbardziej intuicyjna liczba: „na ile meczów z 590 testowych model wskazał właściwego zwycięzcę". To jest liczba, którą warto raportować promotorowi. Test Accuracy 62.46% to liczba dla zbioru po symetryzacji (1180 wierszy), Match Accuracy 61.02% jest dla 590 meczów.

---

## Metryki jakości prawdopodobieństwa

Accuracy mówi tylko czy model trafił/nie trafił. Ale model zwraca PRAWDOPODOBIEŃSTWA (np. 0.73). Czy to prawdopodobieństwo jest dobrze skalibrowane? Do tego służą poniższe metryki.

### Brier Score — u nas 0.2284 (im mniej tym lepiej)
**Co to:** średni kwadrat różnicy między prawdopodobieństwem a wynikiem (0 lub 1).

`Brier = średnia((prawdopodobieństwo - rzeczywista_klasa)²)`

**Skala:** 0 = idealny model (zawsze 1.0 dla y=1, 0.0 dla y=0), 0.25 = model losowy (każda predykcja = 0.5), 1 = najgorszy możliwy (zawsze pewny ale błędny).
**Co znaczy 0.2284?** Trochę lepiej niż losowy (0.25), ale model jest tylko umiarkowanie pewny swoich predykcji.

**Czemu nazwa „Brier"?** Od Glenn W. Brier (meteorolog, 1950) — wymyślił tę metrykę do oceny prognoz pogody.

### Log-loss (logarytmiczna strata) — u nas 0.6464
**Co to:** kara nakładana za pewne ale błędne predykcje.

`Log-loss = -średnia(rzeczywista × log(prob) + (1-rzeczywista) × log(1-prob))`

**Skala:** 0 = idealny, 0.693 = losowy (zgaduje 0.5 dla wszystkiego), wyższy = gorszy. Brak górnej granicy — przewidzieć 0.01 dla y=1 daje karę ~4.6.
**Co znaczy 0.6464?** Lekko lepiej niż losowy (0.693). Model nie jest skrajnie pewny błędnych predykcji.

**Różnica vs Brier:** log-loss MOCNIEJ karze pewne błędy. Brier kara skaluje się liniowo z kwadratem różnicy, log-loss skaluje się logarytmicznie — bardzo pewna błędna predykcja (np. 0.99 zamiast 0) dostaje gigantyczną karę w log-loss, w Brier tylko 0.98.

### ECE (Expected Calibration Error) — u nas 0.0403 (im mniej tym lepiej)
**Co to:** miara KALIBRACJI prawdopodobieństw. Mówi: „o ile średnio model jest pewny inaczej niż w rzeczywistości się sprawdza".

**Jak liczymy:**
1. Dzielimy predykcje na 10 koszyków: [0.0-0.1], [0.1-0.2], ..., [0.9-1.0]
2. Dla każdego koszyka liczymy:
   - średnie przewidziane prawdopodobieństwo (np. 0.75)
   - rzeczywista frakcja wygranych (np. 0.67)
   - różnica = błąd kalibracji (0.75 - 0.67 = 0.08)
3. ECE = ważona średnia tych błędów (waga = ile predykcji wpadło do tego koszyka)

**Co znaczy 0.0403?** Model jest średnio o 4 p.p. „obok" — np. gdy mówi „70% szansy", w rzeczywistości wygrywa 66% lub 74%.

**Po co kalibracja w ogóle?** Bo jeśli stawiasz zakłady na tenisa, potrzebujesz prawdziwego prawdopodobieństwa — nie samego „tak/nie". Model który mówi „99% szansy" gdy w rzeczywistości to 60%, to model niebezpiecznie nadmiernie pewny.

### Reliability Diagram
Wykres: oś X = średnie przewidziane prawdopodobieństwo w koszyku, oś Y = rzeczywista frakcja wygranych. Punkty na linii y=x = idealna kalibracja. Punkty pod linią = model jest nadmiernie pewny. Punkty nad linią = model jest niedostatecznie pewny.

---

## Kalibracja — szczegóły

### Probability Calibration (Platt scaling)
Random Forest słynie z bycia nadmiernie pewnym blisko 0.5 i niedostatecznie pewnym blisko 0 lub 1. Kalibracja Platta uczy małą regresję logistyczną na wyjściu RF, mapując surowe prawdopodobieństwa na lepiej skalibrowane.

**Jak działa:**
1. Trenujemy RF na zbiorze treningowym, dostajemy surowe prawdopodobieństwa
2. Bierzemy zbiór walidacyjny (niewidziany przez RF)
3. Dla każdego meczu mamy: P_raw (z RF) i y_actual (0 lub 1)
4. Trenujemy regresję logistyczną: `P_kalibrowane = sigmoid(a × P_raw + b)`
5. Stosujemy ją do nowych predykcji RF

**Czy zawsze poprawia?** Nie zawsze. U nas Brier raw = 0.2284, calibrated = 0.2284 — bez różnicy. Bo RF z dobrymi hiperparametrami nie jest mocno źle kalibrowany.

### FrozenEstimator
Klasa z sklearn ≥1.8 oznaczająca: „weź ten już-wytrenowany model i nie trenuj go ponownie". Zastępuje przestarzały parametr `cv="prefit"`. Używamy gdy chcemy kalibrować już-wytrenowany RF na osobnym zbiorze walidacyjnym.

### Threshold Tuning
Domyślnie klasyfikator binarny mówi „y=1 gdy P > 0.5". Threshold tuning szuka czy inny próg (np. 0.45 albo 0.55) daje lepszą Match Accuracy na walidacji, i stosuje go na teście.

**U nas:** optymalny prog = 0.50 (czyli bez zmiany). Z teoretycznych względów — w danych symetryzowanych każde inne ustawienie progu „oszukuje" ewaluację. Bezpieczna decyzja: zostać przy 0.5.

---

## Walidacja krzyżowa i podział danych

### Train/Val/Test split (60/20/20)
- **Train (60%)**: na tym model się uczy
- **Val (20%)**: na tym wybieramy hiperparametry i kalibrujemy
- **Test (20%)**: tylko do finalnej oceny

**Czemu chronologiczny?** Pierwsze 60% sezonu = trening, środkowe 20% = walidacja, ostatnie 20% = test. Bez chronologii model widziałby przyszłość → leakage → wyniki by się napompowały.

**Czemu 60/20/20 a nie 80/10/10?** Większy zbiór walidacyjny i testowy = bardziej wiarygodne metryki. Tylko 10% to ~300 meczów, statystyka będzie szumowa.

### TimeSeriesSplit
Cross-validation z zachowaniem porządku czasowego. Zamiast losowo dzielić na fold-y (zwykły KFold), bierze coraz większe początkowe okno jako trening i kolejne okno jako test:
- Iter 1: trening [0:700], test [700:1400]
- Iter 2: trening [0:1400], test [1400:2100]
- ... itd.

Bez tego trening mógłby zawierać marzec 2024, a test wrzesień 2023 — model „pamiętałby" przyszłość.

### Expanding Window (dla cech dynamicznych)
Forma gracza w meczu nr 1500 = średnia jego wyników z meczów 0-1499 (czyli wszystkich PRZED bieżącym). Nigdy nie używamy meczu 1501+ do liczenia formy na mecz 1500 — bo to byłby wyciek danych z przyszłości.

### Cold start
Pierwsze mecze gracza w sezonie 2024 — nie mamy jeszcze nic w historii 2024 do liczenia jego formy. Rozwiązanie: dodajemy dane historyczne 2018-2023 jako bufor. Dzięki temu nawet pierwszy mecz Djokovica w 2024 ma forma policzoną z 7 poprzednich lat.

---

## Pojęcia z Model Slicing

### Slice (warstwa, podgrupa)
Podzbiór meczów spełniających określone warunki (np. `best_of=5 AND round=QF AND surface=Grass`). Model Slicing = systematyczne sprawdzanie jak model działa NA TYCH PODGRUPACH, a nie tylko ogólnie.

**Po co?** Bo ogólne 61% accuracy może ukrywać:
- 70% accuracy na meczach Bo3 (większość, ~82%)
- 40% accuracy na meczach Bo5 (mniejszość, ~18%)
- Średnio wychodzi 64% × 0.82 + 40% × 0.18 = 56% — i można naprawić ten 40%

### Support (wsparcie)
Liczba próbek w slice'ie. Slice z support=3 to praktycznie szum (każdy mecz waży 33%). Slice z support=100 to solidna próbka. U nas filtrujemy slice'y z `MIN_SUPPORT=5`.

### Underperformance Gap
Różnica między accuracy w slice'ie a accuracy ogólnym. Slice z gap=-0.15 to slice gdzie model jest 15 p.p. gorszy niż średnio — kandydat do poprawy.

### Wilson Confidence Interval (Wilson CI)
Sposób na obliczenie 95% przedziału ufności dla proporcji (np. accuracy). Lepszy od klasycznego Wald'a dla małych prób.

**Przykład:** Slice ma 8 meczów, 5 trafionych (accuracy = 62.5%). Wald'owski CI: 0.625 ± 1.96 × √(0.625×0.375/8) = [0.30, 0.96]. Wilson CI: [0.31, 0.86] — sensowniejszy.

**Dlaczego Wilson lepszy?** Bo Wald daje absurdalnie szerokie CI dla skrajnych proporcji (1/8 trafień = [-0.10, 0.35], a powinno być [0.02, 0.47]). Wilson nigdy nie wychodzi poza [0,1] i lepiej radzi sobie z małymi próbkami.

### statistically_below_overall flag
Boolean dla każdego slice'a: czy DOLNA granica jego Wilson CI jest mniejsza od accuracy ogólnego. Jeśli tak — można twierdzić ze 95% pewnością „model jest tu istotnie gorszy", nie tylko „przypadkowo gorszy".

---

## Symetryzacja danych

### Co to jest symetryzacja?
Dane Jeff Sackmana są asymetryczne: kolumny `winner_*` i `loser_*` zawsze opisują zwycięzcę i przegranego. Bezpośrednio użyte = positional bias (model nauczy się że „Gracz 1" zawsze wygrywa).

**Rozwiązanie:** każdy mecz generuje 2 wiersze:
1. Wiersz A: Gracz 1 = winner, Gracz 2 = loser, y=1
2. Wiersz B: Gracz 1 = loser, Gracz 2 = winner, y=0

Symbol cech: `w_rank`/`l_rank` (winner/loser perspective) → `p1_rank`/`p2_rank` (Gracz 1/Gracz 2 — neutralne).

### Cechy różnicowe (diff)
`rank_diff = p1_rank_log - p2_rank_log`. Wartość dodatnia = Gracz 2 wyżej w rankingu. Cechy różnicowe są symetryczne (zamiana p1↔p2 daje znak przeciwny) — ułatwiają modelowi nauczyć się że to RÓŻNICA decyduje, nie wartości absolutne.

---

## Pojęcia z poszczególnych wariantów

### Best of 5 (Bo5) vs Best of 3 (Bo3)
Bo5 = mecz do 3 wygranych setów (5 setów maks). Tylko Grand Slamy. ~18% danych.
Bo3 = do 2 setów (3 setów maks). Wszystko poza GS. ~82% danych.

### Endurance Score
Złożona cecha (wariant Bo5) — kompozyt 6 składników: forma Bo5, forma w długich meczach, doświadczenie Bo5, doświadczenie w długich meczach, średni czas Bo5, stabilność serwisu. Mówi modelowi „jak dobrze gracz znosi dystans 5 setów".

### Seed (rozstawienie)
Pre-tournament ocena gracza przez organizatora. Top seedy są chronieni od siebie w drabince. Cechą jest `seed_context_score` — znormalizowany do [0,1].

### Tournament Path Opponent Strength
W bieżącym turnieju, gracz pokonał już 3 rywali. Liczymy średnią ich siły (log1p z punktów rankingowych). Mówi modelowi: „gracz dotarł do QF łatwą czy trudną drogą".

### Pressure Serve Profile
Profil serwisowy w meczach „pod presją" — definiowanych jako Bo5 LUB późna runda (QF/SF/F na Bo3). Inteligentny wybór: jeśli bieżący mecz Bo5, używamy Bo5 profilu; jeśli późna runda Bo3, używamy late-round profilu.

---

## Ostatnie ważne pojęcia

### Feature Importance
Z RF dostajemy „ważność" każdej cechy — liczba 0-1, jak mocno cecha wpływa na decyzje drzew. Sortujemy malejąco. Pierwsze 10 cech = te które RF najbardziej wykorzystuje.

### Module-level state (pattern w Pythonie)
Globalne zmienne (`_HISTORY_INDEX`, `_HISTORY_CUTOFF`) ustawiane przez funkcję setter (`set_history_context`). Zamiast przekazywać te wartości jako argument do 25 funkcji, ustawiamy raz i wszystkie funkcje widzą. Trade-off: brzydsze (globalny stan), ale dramatycznie czystszy kod.

### Bisect (z modułu `bisect` w Pythonie)
Binary search dla posortowanej listy. `bisect_left(arr, x)` zwraca pozycję, gdzie należałoby wstawić x żeby zachować sortowanie. Używamy do szybkiego wyciągania historii gracza ROZEGRANEJ PRZED danym cutoff'em.

### Monkey patch
Podmiana funkcji bibliotecznej w runtime. W slicecompare.py podmieniamy `runpy.run_path` na wersję z cache'em — globalnie, dla wszystkich pakietów. Pragmatyczne, ale ryzykowne (jeśli inny kod używa runpy, też zostanie wpływnięty).

### CalibratedClassifierCV
Klasa sklearn do kalibracji. Owija inny estymator (np. RF) i dodaje warstwę kalibracji (Platt sigmoid lub isotonic). `method="sigmoid"` = Platt scaling.

### subprocess vs runpy
`subprocess` uruchamia osobny proces Pythona (drogo, ale izolowane). `runpy.run_path` uruchamia skrypt w bieżącym procesie (taniej, ale dzielimy pamięć). Używamy runpy żeby móc cache'ować namespace baseline'u w pamięci.

### namespace (w kontekście runpy)
Słownik wszystkich zmiennych globalnych po wykonaniu skryptu. Wynik `runpy.run_path("main_48_cech.py")` to dict z `df_train_raw`, `best_rf`, `match_accuracy` itd. Pozwala importować wyniki bez modyfikowania skryptu na funkcje.
