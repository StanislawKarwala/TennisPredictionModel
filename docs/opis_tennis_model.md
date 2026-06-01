# Opis pliku `tennis_model.py` — model bazowy

> Wszystkie pojęcia techniczne (CV Accuracy, Brier, log-loss, ECE, kalibracja itd.) są szczegółowo wyjaśnione w `SLOWNICZEK_POJEC.md`. Tu są skrócone wytłumaczenia inline.

## Co ten plik robi w jednym zdaniu
Trenuje model **Random Forest** (las stu drzew decyzyjnych głosujących wspólnie), który przewiduje, który z dwóch tenisistów wygra mecz, na podstawie 40 cech (rankingi, forma, statystyki serwisowe, itd.) i mierzy jego skuteczność na czterech metrykach.

## Główne założenia

1. **Chronologia danych** — model nie może „widzieć przyszłości". Mecze dzielimy po dacie: pierwsze 60% to trening, środkowe 20% to walidacja, ostatnie 20% to test. Tak jak na egzaminie — uczysz się z materiałów ze szkoły, ale sprawdzian dostajesz z nowych zadań.

2. **Historia dla cech dynamicznych** — niektóre cechy (np. „aktualna forma gracza") wymagają znajomości jego ostatnich meczów. Na początku 2024 roku takiej historii nie ma (cold start), więc doczytujemy mecze z lat 2018-2023 jako „pamięć" o graczach.

3. **Symetryzacja** — w surowych danych zawsze „winner" jest jako Gracz 1, a „loser" jako Gracz 2. Gdybyśmy tak trenowali, model nauczyłby się głupiej reguły „Gracz 1 zawsze wygrywa" (positional bias). Rozwiązanie: każdy mecz dajemy modelowi dwa razy — raz z perspektywy zwycięzcy (y=1), raz z perspektywy przegranego (y=0). Wtedy model musi nauczyć się NAPRAWDĘ rozpoznawać kto wygra.

4. **Random Forest** — wybór algorytmu. To „las" wielu prostych drzew decyzyjnych, które razem głosują. Dobrze radzi sobie z danymi tabelarycznymi, jest odporny na overfitting (nadmierne dopasowanie do danych treningowych), nie wymaga skalowania cech.

5. **Walidacja krzyżowa chronologiczna (TimeSeriesSplit)** — żeby uczciwie ocenić hiperparametry. Zwykła walidacja krzyżowa losowo dzieli na 5 fragmentów; my dzielimy chronologicznie, żeby trening w każdym fold-zie był ZAWSZE przed testem czasowo.

## Dlaczego takie wybory, a nie inne?

- **Czemu Random Forest, a nie sieć neuronowa?** Mamy tylko ~3000 meczów rocznie. Sieci neuronowe potrzebują dziesiątek tysięcy przykładów żeby świecić — Random Forest działa świetnie na małych zbiorach tabelarycznych i sam znajduje ważne cechy.
- **Czemu log z rankingu, a nie surowy ranking?** Różnica między rankiem 1 a 10 to przepaść (Djokovic vs ktoś z top 10). Różnica między 100 a 110 to prawie nic. Logarytm „spłaszcza" te duże liczby na końcu, dzięki czemu model lepiej rozumie skalę.
- **Czemu TimeSeriesSplit, a nie losowy KFold?** Mecze są szeregiem czasowym. Losowy podział spowodowałby, że model „widziałby" mecze z przyszłości w fazie walidacji — to byłoby oszustwo (data leakage).
- **Czemu RandomizedSearchCV (50 losowych prób), a nie GridSearchCV (wszystkie kombinacje)?** Grid jest wykładniczo wolny. Losowa próbka 50 zestawów hiperparametrów daje >95% szansy że trafimy w okolice optimum przy znacznie mniejszym koszcie.
- **Czemu próg 0.5 a nie inny (threshold tuning)?** Po symetryzacji każdy mecz ma dwie perspektywy o sumie prawdopodobieństw 1. Naturalnym progiem podziału jest 0.5. Próbowaliśmy threshold tuningu — okazał się degenerate (zawsze podbijał próg do skrajności bo „oszukiwał" na danych symetryzowanych).
- **Czemu kalibracja Platta (sigmoid), a nie isotonic?** Platt zakłada że krzywa kalibracji to sigmoid (gładka funkcja S) — pasuje dla małych zbiorów (590 meczów walidacyjnych). Isotonic wymaga więcej danych, na małych próbkach przeucza się.

## Słowniczek metryk z tego pliku (skrót — szczegóły w SLOWNICZEK_POJEC.md)

| Metryka | Co to mierzy | U nas | Idealne |
|---|---|---|---|
| **CV Accuracy** | trafność na zbiorze treningowym dzielonym na 5 fold-ów (kontrolne porównanie hiperparametrów) | 64.78% | 100% |
| **Validation Accuracy** | trafność na zbiorze walidacyjnym (20% — sanity check po wyborze hiperparametrów) | 63.31% | 100% |
| **Test Accuracy** | trafność na zbiorze testowym (20% — finalny test, na danych symetryzowanych) | 61.36% | 100% |
| **Match Accuracy** | trafność na poziomie MECZU (% poprawnie przewidzianych zwycięzców z 590 meczów testowych) | **61.02%** | 100% |
| **Brier Score** | średni kwadrat różnicy prawdopodobieństwo–wynik (jak źle są skalibrowane probabilities) | 0.2284 | 0 |
| **log-loss** | kara za pewne ale błędne predykcje (rośnie logarytmicznie gdy model jest pewny i myli się) | 0.6464 | 0 |
| **ECE** | Expected Calibration Error — średnie odchylenie między „model mówi 70%" a „faktycznie wygrywa 67%" | 0.0403 | 0 |

**Dlaczego 4 różne accuracy?** Bo każde mierzy co innego:
- CV Acc = porównanie hiperparametrów (decyzja: które są najlepsze)
- Validation Acc = sanity check po wyborze hiperparametrów
- Test Acc = finalna ocena uczciwa
- Match Acc = praktyczna „% trafionych meczów" — to liczba do raportu

## Ważne metody (1:1 z kodu)

| Metoda | Co robi |
|---|---|
| `calculate_form(player_name, history)` | Liczy bieżącą formę gracza jako % zwycięstw z ostatnich 10 meczów (rolling window). Jak gracz nigdy nie grał — zwraca 0.5 (neutralne). |
| `get_h2h(p1, p2, history)` | Bilans bezpośrednich pojedynków (head-to-head): ile razy p1 wygrał z p2 minus odwrotnie. Pokazuje, kto „ma" rywala. |
| `calculate_surface_form(player_name, surface, history)` | Forma gracza, ale tylko na danej nawierzchni (np. Nadal na clay vs Nadal na grass). |
| `calculate_serve_stats(player_name, history, window=10)` | Wyciąga 8 statystyk serwisu z ostatnich meczów: ile asów na serwis, % wygranych pierwszych serwisów, % obronionych break pointów itd. Te liczby mówią modelowi „jak silnie serwuje" + „jak dobrze returnuje". |
| `_build_player_index(full_sequence)` | Wstępnie buduje mapę „nazwa gracza → numery wierszy gdzie wystąpił". Optymalizacja: zamiast szukać meczów gracza przez przejrzenie całej historii (18 000 wierszy), patrzymy od razu w listę indeksów. Bisect dla cutoff w O(log K). |
| `add_dynamic_features(df_subset, historical_data)` | Główna pętla, która do każdego meczu dolicza wszystkie cechy dynamiczne (form, h2h, surface form, serve stats). **Bardzo ważne**: dla i-tego meczu jako historię traktujemy TYLKO mecze i-1, i-2, … (expanding window — żeby nie podglądać przyszłości). |
| `symmetrize_data(df_subset, shuffle)` | Tworzy z każdego meczu dwa wiersze (zwycięzca jako p1 / przegrany jako p1). Eliminuje obciążenie pozycyjne (positional bias). Wymaga shuffle żeby model nie nauczył się że „co drugi wiersz to wygrana". |
| `select_match_level_threshold(...)` | **NIE robi tuningu** w wersji finalnej. Zwraca stały próg 0.5 bo dla symetrycznych danych każdy inny próg jest gamingiem ewaluacji (poprzednia wersja wycieniała próg do skrajności i dawała fałszywe 93%). |
| `apply_match_level_threshold(test_data, threshold)` | Stosuje próg do prawdopodobieństw modelu i liczy match accuracy. Match accuracy = na ile meczów z 590 model wskazał właściwego zwycięzcę. |
| `compute_reliability_table(y_true, y_proba)` | Sprawdza kalibrację: dzieli predykcje na 10 koszyków [0.0-0.1, 0.1-0.2, ...], dla każdego liczy średnie predykowane prawdopodobieństwo vs rzeczywistą frakcję wygranych. Jak model jest dobrze skalibrowany, te dwie liczby są blisko siebie. |
| `save_reliability_diagram(table, path, title)` | Rysuje wykres kalibracji (oś X=prawdopodobieństwo, oś Y=rzeczywista frakcja). Punkty na linii y=x = idealna kalibracja. Zapisuje PNG do `reports/outputs/reliability_diagram.png`. |
| `evaluate_calibration_quality(y_true, y_proba)` | Liczy 3 miary jakości prawdopodobieństw naraz: Brier (im niższy tym lepszy, max 1), log-loss (im niższy tym lepszy, brak górnej granicy), ECE (Expected Calibration Error w [0,1]). |

## Ważne zmienne

| Zmienna | Co oznacza |
|---|---|
| `RANDOM_STATE` | Ziarno losowości. Ustawione na 42 (albo z env var TENNIS_RANDOM_STATE), żeby wyniki były powtarzalne między uruchomieniami. |
| `BASE_DIR`, `DATA_DIR`, `OUTPUTS_DIR` | Ścieżki bazujące na lokalizacji pliku. Niezależne od cwd. |
| `df_base` | Wszystkie mecze 2024 po wstępnym czyszczeniu (40 cech + winner/loser names). |
| `df_history_base` | Mecze z lat 2018-2023, tylko jako kontekst historyczny. Model NIE jest na nich trenowany, ale używamy ich do liczenia form i H2H. |
| `df_train_raw`, `df_val_raw`, `df_test_raw` | Trzy chronologiczne podziały: 60/20/20 z roku 2024 (~1770/590/590 meczów). |
| `cols_base` | Lista kolumn surowych z CSV (nawierzchnia, ranking, statystyki serwisu, nazwy graczy). |
| `cols_serve` | 18 kolumn ze statystykami serwisowymi (9 dla winnera, 9 dla losera). |
| `SERVE_DEFAULTS` | Słownik wartości domyślnych dla statystyk serwisu — średnie tourowe ATP. Używane jako fallback gdy gracz nie ma jeszcze historii. |
| `param_dist` | Słownik z wszystkimi wariantami hiperparametrów dla RandomizedSearchCV (liczba drzew, głębokość, min próbek do podziału itd.). |
| `search` | Obiekt RandomizedSearchCV — sprawdza 50 losowych kombinacji hiperparametrów na 5-fold TimeSeriesSplit i wybiera najlepszą po CV Accuracy. |
| `best_rf` | Wytrenowany Random Forest z najlepszymi znalezionymi hiperparametrami. |
| `val_acc`, `test_acc` | Validation Accuracy / Test Accuracy (na symetryzowanych danych — czyli binarna klasyfikacja, 1180 wierszy). |
| `match_accuracy` | **KLUCZOWA METRYKA** — accuracy na poziomie meczu (% poprawnie przewidzianych zwycięzców z 590 meczów testowych). U mnie wyszło **61.02%**. |
| `winner_perspective` | Tabela z testu, gdzie p1 jest zawsze faktycznym zwycięzcą — używana do liczenia match_accuracy. Modyfikacja: bierzemy tylko jedną z dwóch perspektyw symetryzowanych. |
| `calibrator` | Model do kalibracji prawdopodobieństw (Platt scaling przez CalibratedClassifierCV). Dopina logistic regression do wyjścia Random Forest. |
| `match_accuracy_tuned` | Match accuracy po kalibracji prawdopodobieństw (u mnie 60.68% — kalibracja nieznacznie pogorszyła, bo RF jest już dość dobrze skalibrowany). |
| `feature_importance` | Tabela mówiąca, które cechy najmocniej wpływają na decyzje modelu (Gini importance z RF). Najwyżej zwykle: `rank_diff`, `p1_rank_log`, `p2_rank_log`, `form_diff`. |
| `reliability_table` | Tabela kalibracji 10 koszyków × kolumny [bin_lower, bin_upper, count, avg_predicted, observed_frequency, calibration_error]. |

## Wyniki

| Metryka | Wartość | Co znaczy w skrócie |
|---|---|---|
| CV Accuracy | **64.78%** | Średnia z 5 fold-ów na treningu (do wyboru hiperparametrów) |
| Validation Accuracy | **63.31%** | Na zbiorze walidacyjnym (sanity check) |
| Test Accuracy (binary) | **61.36%** | Na symetryzowanym teście (1180 wierszy) |
| **Match Accuracy** | **61.02%** | Na 590 meczach — to jest TA liczba do raportu |
| Przewaga vs losowy | +11.0 p.p. | Random guessing = 50%, nasz model = 61% |
| Brier Score | 0.2284 | Lekko lepsze niż losowe (0.25). Model nie jest bardzo pewny |
| log-loss | 0.6464 | Lekko lepsze niż losowe (0.693) |
| ECE | 0.0403 | Średnio o 4 p.p. „obok" — dobra kalibracja |

## Co odpowiedzieć gdy promotor zapyta…

**Q: „Czemu Random Forest, a nie sieć neuronowa / XGBoost?"**
A: Random Forest jest najprostszym sensownym wyborem dla danych tabelarycznych o tej skali (~3000 meczów). Sieci neuronowe potrzebują 10-100x więcej danych. XGBoost mógłby dać +1-2 p.p., ale jest trudniejszy do strojenia hiperparametrów i mniej interpretowalny. Wybraliśmy RF dla prostoty i czytelności.

**Q: „Dlaczego accuracy spada między CV (64.78%), validation (63.31%) i test (61.36%)?"**
A: To normalne. CV widzi mecze ze środka sezonu — model już zna te dane częściowo. Validation to świeże 20%. Test to najnowsze 20% — najbardziej różni się od treningu (nowi gracze, kontuzje, transfer formy). Spadek o 1-2 p.p. jest oczekiwany i zdrowy.

**Q: „Dlaczego Match Accuracy (61.02%) jest niższe niż Test Accuracy (61.36%)?"**
A: Bo to różne miary. Test Accuracy patrzy na 1180 wierszy (mecz × 2 perspektywy). Match Accuracy patrzy na 590 meczów. Mecz jest poprawnie przewidziany TYLKO gdy obie perspektywy zgadzają się ze sobą (P(y=1)>0.5 dla winner, P(y=1)<0.5 dla loser). Ta surowsza definicja daje niższą liczbę.

**Q: „Co to jest Brier Score 0.2284?"**
A: To miara jakości prawdopodobieństw. Jeśli model mówi „70% szansy" i gracz wygrał, kara to (1-0.7)²=0.09. Średnia tych kar po wszystkich meczach = Brier. Im niżej tym lepiej. 0.25 = losowy model (każdy mówi 0.5), 0 = idealny. Nasze 0.2284 to lekko lepsze niż losowy — model nie jest super pewny swoich predykcji.

**Q: „Co to jest log-loss 0.6464?"**
A: To inna miara, bardziej karząca pewne ale błędne predykcje. Wzór: -średnia(y×log(p) + (1-y)×log(1-p)). Losowy model = ln(2) ≈ 0.693. Nasze 0.6464 jest lekko niżej = lekko lepiej niż losowy. Brak górnej granicy — model który zawsze mówi „0.99 że Gracz 1" i się myli dostaje karę 4.6 za każdy mecz.

**Q: „Co to jest ECE 0.0403?"**
A: Expected Calibration Error. Dzielimy predykcje na 10 koszyków po 10 p.p. Dla każdego sprawdzamy: czy „średnie przewidziane prawdopodobieństwo" pasuje do „rzeczywistej frakcji wygranych" w tym koszyku? Średnie różnice = ECE. Nasze 4 p.p. znaczy „model przeciętnie pomyli się o 4 p.p. w swojej pewności" — to dobra kalibracja.

**Q: „Co to jest reliability diagram?"**
A: Wykres jakości prawdopodobieństw. Oś X = co model mówi (0 do 1), oś Y = jak często faktycznie się to spełnia. Punkty na przekątnej = idealna kalibracja. Pod przekątną = model nadmiernie pewny (mówi 70%, faktycznie 60%). Nad = niedostatecznie pewny. Zapisywany jako PNG do `reports/outputs/`.

**Q: „Czemu nie zmieniacie progu 0.5 (threshold tuning)?"**
A: Próbowaliśmy. Dla danych symetryzowanych każdy próg inny niż 0.5 łamie symetrię — np. próg 0.4 spowoduje że oba wiersze tego samego meczu mogą predykować to samo (oba „y=1"). To technicznie podnosi accuracy, ale jest gamingiem ewaluacji. Zostawiamy 0.5 jako uczciwy próg.

**Q: „Co to jest kalibracja Platta?"**
A: Mała regresja logistyczna wytrenowana na wyjściu Random Forest, mapująca surowe prawdopodobieństwa na lepiej skalibrowane. Wzór: `P_kal = sigmoid(a × P_raw + b)`. U nas kalibracja nie poprawiła Brier (RF już jest OK), ale jest standardową praktyką w klasyfikatorach probabilistycznych.

**Q: „Co to jest FrozenEstimator?"**
A: Nowy mechanizm z sklearn 1.8+. Mówi „nie trenuj tego modelu jeszcze raz" gdy kalibrujesz go na osobnym zbiorze walidacyjnym. Zastępuje przestarzały `cv="prefit"`.

**Q: „Czemu 40 cech, a nie 100?"**
A: Eksperymentowaliśmy. Dodawanie więcej cech (variant Bo5 ma 77 cech) marginalnie poprawia accuracy ale komplikuje model. 40 to dobry kompromis dla baseline'u — wszystkie cechy mają domenowe uzasadnienie tenisowe.

**Q: „Co znaczy 61% accuracy w kontekście tenisa?"**
A: Random guessing = 50%. Bukmacherskie modele profesjonalne (np. ATP/PINNACLE) osiągają ~65-68%. Akademickie modele Random Forest na ATP osiągają 60-65%. Nasze 61% jest w zakresie akademickim, lekko poniżej state-of-the-art ale zdrowo powyżej losowego. Wskazówki na poprawę: dodać cechy seedingu, drabinki, motywacji.
