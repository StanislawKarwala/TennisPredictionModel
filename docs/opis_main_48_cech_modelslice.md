# Opis pliku `main_48_cech_modelslice.py` — diagnostyka modelu (Model Slicing)

> Wszystkie pojęcia techniczne (Wilson CI, Brier, log-loss, slice, support itd.) są szczegółowo wyjaśnione w `SLOWNICZEK_POJEC.md`. Tu są skrócone wytłumaczenia inline.

## Co ten plik robi w jednym zdaniu
Bierze gotowy model z `main_48_cech.py` i sprawdza, w których konkretnych typach meczów (np. „mecze Bo5", „ćwierćfinały", „leworęczny vs praworęczny") model się myli częściej niż średnio — żeby wiedzieć, co poprawiać.

## Główne założenia

1. **Model nie myli się losowo** — błędy mają strukturę. Może być, że model świetnie typuje mecze na hard court, a zawala Grand Slamy. Bez analizy slice'ów tego nie zobaczymy, bo średnia ogólna (~61%) maskuje różnice między podgrupami.

2. **Slice** = podgrupa meczów z testowego zbioru, wybrana po jednym lub dwóch atrybutach. Np. „wszystkie mecze gdzie best_of=5" albo „wszystkie mecze na surface=Grass & round=QF". Po polsku: warstwa, plasterek, podgrupa.

3. **Pomysł pochodzi z artykułu naukowego** — `GuideAI25_2.pdf` („Model Slicing for Responsible AI", VLDB 2025). Implementacja jest pragmatyczna wersja **SliceFinder** opisanego w tym artykule (https://github.com/yeounoh/slicefinder).

4. **Wilson Confidence Interval** dla odróżnienia prawdziwej słabości od szumu. Jak slice ma support=5 i accuracy=40%, to nie wiemy czy to model jest słaby czy po prostu pech. Wilson CI mówi: „95% pewności, że prawdziwe accuracy leży w przedziale [X, Y]". Jeśli górny brzeg Y jest poniżej średniej, to dopiero wtedy mamy pewność że slice jest istotnie gorszy.

5. **Trzy metryki na slice, nie jedna** — accuracy, Brier, log-loss. Razem dają pełny obraz. Accuracy mówi „trafił czy nie", Brier i log-loss mówią „jak pewny był i czy ta pewność była zasłużona".

## Dlaczego takie wybory, a nie inne?

- **Czemu degree 1 i 2 a nie 3+?** Pełen data cube (kombinacje 8 atrybutów do siebie) miałby tysiące slice'ów, z których większość miałaby support=1-2 (bezużyteczne). Degree 2 to dobry kompromis: wyłapuje interakcje (np. `Grass × QF`) bez kombinatorycznej eksplozji.
- **Czemu MIN_SUPPORT=5?** Slice z 1-4 meczami daje accuracy z absurdalną wariancją (np. 4/4 to „100%" ale nic nie znaczy). 5 to minimum poniżej którego nawet Wilson CI niewiele pomaga.
- **Czemu Wilson, a nie standard CI (Wald)?** Standard Wald `p ± z·√(p(1-p)/n)` totalnie się sypie przy p=0 lub p=1 (zwraca [0,0] lub [1,1]) i przy małych n może dać granice <0 lub >1. Wilson działa zawsze i mieści się w [0,1].
- **Czemu Brier i log-loss?** Accuracy mówi tylko „czy model trafił", ale nie „czy był pewny tej trafnej odpowiedzi". Brier i log-loss karzą model za pewne złe predykcje (np. 95% szans na A, a wygrał B → wielki Brier).
- **Czemu `statistically_below_overall` flag?** Bo `gap < -5p.p.` to za miękkie. Z Wilsonem możemy powiedzieć „95% pewności że slice jest gorszy" — to znacznie mocniejsza twierdzenie.

## Słowniczek pojęć z tego pliku (skrót — szczegóły w SLOWNICZEK_POJEC.md)

| Pojęcie | Co znaczy | Przykład |
|---|---|---|
| **slice** | Podgrupa meczów z testowego zbioru wybrana po atrybutach | „wszystkie mecze gdzie best_of=5 i surface=Grass" |
| **degree** | Liczba atrybutów w slice'ie | degree=1: „best_of=5". degree=2: „best_of=5 & round=QF" |
| **support** | Liczba meczów w slice'ie | Jeśli 12 meczów Bo5 na Grassie, support=12 |
| **gap** | Różnica między accuracy slice'a a accuracy ogólnym | Slice 45%, ogólne 61% → gap=-0.16 (16 p.p. gorszy) |
| **Wilson CI** | 95% przedział ufności dla proporcji | Slice 5/8 = 0.625, CI = [0.31, 0.86] |
| **Wald CI** | Stary klasyczny przedział ufności | Dla 0/4 daje [-0.1, 0.1] — bez sensu |
| **statistically_below_overall** | Flag: czy DOLNY brzeg Wilson CI > accuracy ogólne | True = naprawdę słaby, False = może być szum |
| **Brier Score** | Średni kwadrat (prawdopodobieństwo - prawda)² | Im niżej tym lepiej |
| **log-loss** | Logarytmiczna kara za pewne błędne predykcje | Im niżej tym lepiej |

## Ważne metody (1:1 z kodu)

| Metoda | Co robi |
|---|---|
| `execute_base_pipeline()` | Uruchamia `main_48_cech.py` przez `runpy.run_path` i wyciąga z niego namespace (słownik wszystkich zmiennych po wykonaniu). Dzięki temu reużywamy baseline bez kopiowania kodu — dostajemy `df_test_raw`, `winner_perspective`, `match_accuracy` itd. |
| `require_namespace_items(namespace, names)` | Asercja że baseline zwrócił wszystkie potrzebne zmienne. Jeśli brak, błąd. |
| `prepare_match_level_slice_frame(namespace)` | Buduje tabelę „jeden wiersz = jeden mecz testowy" z atrybutami do slicingu i wynikiem predykcji. Łączy `df_test_raw` z `winner_perspective` po `match_id`. Sprawdza assertion: accuracy po joinie musi się zgadzać z reported_match_accuracy — sanity check. |
| `build_handedness_matchup(row)` | Tworzy etykietę typu „L-vs-R" albo „R-vs-R" (sortuje, żeby L-vs-R i R-vs-L było tym samym slicem — bez tego mielibyśmy dwa identyczne slice'y o innej kolejności liter). |
| `build_bucketed_feature(series, bins, labels)` | Dyskretyzuje ciągłą zmienną na koszyki. Np. różnica rankingowa 87 wpada do koszyka „51-100". Slicing wymaga zmiennych kategorycznych, więc liczby trzeba kubełkować. |
| `wilson_confidence_interval(successes, n, z=1.96)` | Liczy 95% przedział ufności metodą Wilsona dla proporcji. Wzór: `(p + z²/(2n) ± z·√(p(1-p)/n + z²/(4n²))) / (1 + z²/n)`. Zwraca (dolny_brzeg, górny_brzeg) — wartości zawsze w [0,1]. |
| `compute_model_slices(match_slice_frame, slice_columns, ...)` | Główna funkcja — iteruje po wszystkich kombinacjach atrybutów (degree 1 i 2), grupuje mecze, liczy 9 statystyk dla każdej grupy: support, accuracy, gap, lower_ci, upper_ci, brier_score, log_loss, avg_probability, statistically_below_overall. Filtruje grupy z support < MIN_SUPPORT. |
| `slice_description(columns, values)` | Formatuje opis slice'a do czytelnego stringa typu `"best_of=5 & round=QF"`. |
| `print_slice_table(title, slices_df, top_n, gap_threshold)` | Drukuje TOP N najsłabszych slice'ów (gdzie gap_threshold=-0.05, czyli minimum 5 p.p. gorzej niż średnia). Sortuje po `gap` rosnąco — najsłabsze na górze. |
| `print_best_slices(slices_df, top_n)` | Drukuje TOP N NAJLEPSZYCH slice'ów — dla kontrastu, żeby zobaczyć gdzie model działa najlepiej. |
| `run_model_slicing()` | Funkcja main — uruchamia całość: baseline → tabela → liczenie → drukowanie. |

## Ważne zmienne

| Zmienna | Co oznacza |
|---|---|
| `BASE_SCRIPT` | Ścieżka do `main_48_cech.py` (domyślnie, można nadpisać env var TENNIS_MODEL_SLICE_BASE). |
| `MIN_SUPPORT = 5` | Minimalna liczba meczów w slice'ie żeby go w ogóle uwzględnić. Chroni przed niewiarygodnymi statystykami z małych próbek. |
| `MAX_SLICE_DEGREE = 2` | Maksymalna liczba atrybutów w jednym slice'ie. 1 = pojedyncze atrybuty, 2 = pary atrybutów. |
| `UNDERPERFORMANCE_GAP = -0.05` | Próg „znaczącej słabości" — slice musi być co najmniej 5 p.p. gorszy od średniej, żeby trafić do raportu. |
| `TOP_N = 12` | Ile najsłabszych slice'ów wyświetlamy w raporcie. |
| `WILSON_Z = 1.96` | Z-score dla 95% przedziału ufności (standard statystyczny — odpowiada dwóm odchyleniom standardowym normalnej). |
| `LOG_EPS = 1e-15` | Zabezpieczenie przed log(0) w liczeniu log-loss (log(0) = -nieskończoność). |
| `slice_columns` | Lista 8 atrybutów do slicingu: surface, tourney_level, best_of, round, handedness_matchup, rank_gap_bucket, age_gap_bucket, form_gap_bucket. |
| `slice_results` | Główna tabela wynikowa — jeden wiersz na slice z 9 kolumnami metryk i statystyk istotności. |
| `match_slice_frame` | Tabela jeden-wiersz-na-mecz z atrybutami slicingu i predykcją. Wejście do `compute_model_slices`. |

## Co wyciągnęliśmy z tego

**Najsłabsze konkretne podgrupy** (po uruchomieniu):
- `best_of=5` — Grand Slamy, accuracy ~48-52% (vs średnia 61%)
- `round=R128` — pierwsza runda GS, accuracy bywa 30-40% w niektórych subgrupach
- `handedness_matchup=L-vs-R` — leworęczny vs praworęczny, accuracy 40-50% w niektórych
- `round=F` — finały (mała próbka, więc CI szeroki — często NIE jest istotnie gorszy)

**Najlepsze podgrupy**: zwykle te z dużą różnicą rankingu (`rank_gap=>100`) — łatwo przewidzieć faworyta.

**Następny krok**: te insighty napędzają warianty `sliceaware_*`, które dodają cechy specyficzne dla tych słabych grup.

## Co odpowiedzieć gdy promotor zapyta…

**Q: „Co to jest 'slice' i po co liczyć metryki per slice?"**
A: Slice to podgrupa meczów wybrana po atrybutach (np. „wszystkie mecze Bo5 na Grassie"). Ogólne 61% accuracy może ukrywać 70% dla łatwych meczów i 40% dla trudnych. Slicing pokazuje, gdzie model ma realny problem i co warto poprawić.

**Q: „Co to jest support i czemu minimalny 5?"**
A: Support to liczba meczów w slice'ie. Slice z 2 meczami i accuracy=100% nic nie znaczy (mógł trafić przypadkiem). 5 to minimum żeby statystyka miała choćby cień sensu. Profesjonalne narzędzia (np. SliceFinder z artykułu) używają nawet 10-20.

**Q: „Co to jest Wilson Confidence Interval?"**
A: Przedział, w którym z 95% pewnością znajduje się PRAWDZIWE accuracy slice'a (gdybyśmy mieli nieskończenie dużo meczów tego typu). Np. slice 8 meczów / 5 trafionych → accuracy=62.5%, ale Wilson CI=[31%, 86%]. Czyli prawdziwe accuracy może być od 31% do 86% — bardzo niepewne.

**Q: „Dlaczego Wilson zamiast prostszego (p ± 1.96·SE)?"**
A: Bo standardowy (Wald) sypie się dla małych prób lub skrajnych proporcji. Slice 4 mecze / 0 trafień → accuracy=0%, Wald CI=[0,0] (zerowy przedział, bez sensu). Wilson CI=[0%, 49%] — adekwatnie szeroki. Wilson też nigdy nie wychodzi poza [0,1].

**Q: „Czemu trzy metryki na slice (accuracy + Brier + log-loss), a nie tylko accuracy?"**
A: Accuracy mówi „czy model trafił" ale nie „czy był pewny". Slice gdzie model trafia 60% z prawdopodobieństwami 0.51-0.55 to inny przypadek niż 60% z prawdopodobieństwami 0.30 i 0.95 (drugi to model który myli się BARDZO PEWNYCH). Brier i log-loss łapią różnicę.

**Q: „Co to jest `statistically_below_overall`?"**
A: Flag w wynikowej tabeli mówiąca: „czy nawet OPTYMISTYCZNY szacunek (górny brzeg Wilson CI) tego slice'a jest niższy niż średnia ogólna?". Jeśli True, możemy ze statystyczną pewnością powiedzieć „model jest tu istotnie gorszy". Bez tej flagi „slice jest 5 p.p. gorszy" może być po prostu szumem.

**Q: „Skąd te 8 atrybutów slicingu (surface, level, best_of, round, handedness, rank_gap, age_gap, form_gap)?"**
A: To są wszystkie semantyczne wymiary meczu tenisowego, które gracz/komentator by wymienił charakteryzując mecz: rodzaj kortu, ranga turnieju, format setów, etap, dominująca ręka, dystans rankingowy, dystans wiekowy, dystans formy. Każdy z nich może być przyczyną innego zachowania modelu.

**Q: „Czemu degree=2 a nie 3?"**
A: Z 8 atrybutów: degree 1 daje 8 slice'ów. Degree 2 daje C(8,2)=28 par. Degree 3 dałoby 56 trójek, ale większość miałaby support<5 (z 590 testowych meczów rozcieńczamy na trójkach). Dwóki łapią najważniejsze interakcje bez kombinatorycznej eksplozji.

**Q: „Co dalej z tymi insighty?"**
A: Wskazują, gdzie dodać cechy domenowe. Słaby slice „Bo5" → wariant `sliceaware_bestof5_v1` z cechami wytrzymałości. Słaby slice „QF" → wariant `sliceaware_qfserve_v3` z cechami drabinki. Slicing to narzędzie diagnostyczne — same liczby nie poprawiają modelu, ale wskazują kierunek interwencji.
