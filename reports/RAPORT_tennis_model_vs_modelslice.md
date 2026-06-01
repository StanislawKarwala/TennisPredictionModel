# Raport porównawczy: `main_48_cech.py` vs `main_48_cech_modelslice.py`

## 1. Cel raportu

Ten dokument ma w prosty, ale dokładny sposób wyjaśnić:

1. co robi podstawowy model z pliku `main_48_cech.py`,
2. co dodatkowo robi plik `main_48_cech_modelslice.py`,
3. czy `main_48_cech_modelslice.py` zmienia sam model,
4. jakie są wyniki modelu,
5. co wynika z analizy `model slicing`,
6. jak można to sensownie przedstawić promotorowi.

Ten raport został przygotowany na podstawie rzeczywistego uruchomienia obu plików w bieżącym środowisku projektu.


## 2. Najkrótsza odpowiedź

Najważniejszy wniosek jest taki:

`main_48_cech_modelslice.py` nie buduje innego modelu niż `main_48_cech.py`.

On:

1. uruchamia dokładnie ten sam pipeline treningu i testu,
2. dostaje dokładnie te same predykcje,
3. a dopiero potem robi dodatkową analizę: sprawdza, dla jakich grup meczów model działa gorzej albo lepiej.

Czyli:

1. `main_48_cech.py` odpowiada na pytanie: "jak dobry jest model ogólnie?"
2. `main_48_cech_modelslice.py` odpowiada na pytanie: "na jakich typach meczów ten model działa słabo, a na jakich dobrze?"


## 3. Co robi `main_48_cech.py`

Plik `main_48_cech.py` buduje klasyczny model predykcyjny dla meczów ATP.

W uproszczeniu działa to tak:

1. wczytuje dane z sezonu 2024,
2. dołącza historię z lat 2018–2023,
3. tworzy cechy statyczne i dynamiczne,
4. symetryzuje dane, czyli z jednego meczu robi dwie próbki,
5. uczy model `RandomForestClassifier`,
6. ocenia model na walidacji, teście i na poziomie rzeczywistych zwycięzców meczów.

Model korzysta z 40 cech wejściowych. Są tam między innymi:

1. kontekst meczu: nawierzchnia, poziom turnieju, `best_of`, runda,
2. cechy rankingowe: ranking ATP i punkty rankingowe,
3. cechy gracza: wiek, wzrost, ręczność,
4. forma ogólna,
5. forma na nawierzchni,
6. H2H,
7. rolling statystyki serwisu i returnu,
8. cechy różnicowe między graczami.


## 4. Co robi `main_48_cech_modelslice.py`

Ten plik nie uczy nowego modelu. On uruchamia bazowy model i dopina warstwę diagnostyczną.

Najprościej:

1. bierze gotowy model i jego wyniki,
2. patrzy na prawdziwe mecze testowe,
3. dzieli je na sensowne grupy,
4. sprawdza accuracy osobno dla każdej grupy,
5. porównuje tę accuracy z accuracy ogólnym.

To właśnie jest `model slicing`.


## 5. Co to jest `slice` bardzo prostymi słowami

`Slice` to po prostu wycinek danych, czyli jakaś konkretna grupa meczów.

Przykłady:

1. wszystkie mecze `Best of 5`,
2. wszystkie mecze w rundzie `QF`,
3. wszystkie mecze `L-vs-R`, czyli leworęczny kontra praworęczny,
4. wszystkie mecze, gdzie różnica rankingu jest duża,
5. wszystkie mecze na danej nawierzchni.

To można porównać do badania klas uczniów.

Jeżeli nauczyciel powie:

"Średni wynik całej szkoły to 61%"

to jeszcze nie wiadomo:

1. czy każda klasa radzi sobie podobnie,
2. czy może jedna klasa ma 80%, a inna 40%.

`Model slicing` robi właśnie coś takiego dla modelu.

Nie pyta tylko:

"jaki jest wynik modelu ogólnie?"

ale też:

"czy model nie psuje się mocno dla konkretnego typu meczów?"


## 6. Jak działa slicing w tym projekcie

W tym projekcie slicing działa na poziomie rzeczywistych meczów, a nie na poziomie sztucznie podwojonych próbek po symetryzacji.

To bardzo ważne.

Dlaczego?

Bo model jest trenowany na danych symetryzowanych, ale człowieka interesuje finalnie wynik prawdziwego meczu.

Dlatego `main_48_cech_modelslice.py`:

1. bierze `df_test_raw`, czyli prawdziwe mecze testowe,
2. bierze `winner_perspective`, czyli ocenę z perspektywy rzeczywistego zwycięzcy,
3. scala to po `match_id`,
4. tworzy jedną tabelę: jeden wiersz = jeden realny mecz,
5. dopiero wtedy liczy accuracy dla grup meczów.

To znaczy, że analiza slice’ów jest uczciwa i łatwa do interpretacji.


## 7. Jakie grupy są analizowane

W obecnej wersji analizowane są grupy zbudowane z następujących atrybutów:

1. `surface`,
2. `tourney_level`,
3. `best_of`,
4. `round`,
5. `handedness_matchup`,
6. `rank_gap_bucket`,
7. `age_gap_bucket`,
8. `form_gap_bucket`.

Analiza obejmuje:

1. slice’y 1D, czyli pojedyncze grupy, na przykład tylko `round=QF`,
2. slice’y 2D, czyli przecięcia dwóch warunków, na przykład `best_of=5` i `rank_gap_bucket=51-100`.

W kodzie ustawiono:

1. minimalny support = 5 meczów,
2. maksymalny stopień slice’a = 2D.

To jest sensowne, bo chroni przed nadinterpretacją bardzo małych grup.


## 8. Porównanie wyników obu plików

### 8.1. Wyniki `main_48_cech.py`

Po uruchomieniu bazowego modelu otrzymano:

1. `CV Accuracy`: `0.6478`
2. `Validation Accuracy`: `0.6331`
3. `Test Accuracy`: `0.6136`
4. `Match Prediction Accuracy`: `0.6102`

Oznacza to, że model poprawnie przewidział zwycięzcę w `360 z 590` meczów testowych.


### 8.2. Wyniki `main_48_cech_modelslice.py`

Po uruchomieniu `main_48_cech_modelslice.py` otrzymano dokładnie te same wyniki modelu:

1. `CV Accuracy`: `0.6478`
2. `Validation Accuracy`: `0.6331`
3. `Test Accuracy`: `0.6136`
4. `Match Prediction Accuracy`: `0.6102`


### 8.3. Wniosek z porównania

To oznacza, że:

1. `main_48_cech_modelslice.py` nie zmienia modelu,
2. nie poprawia go i nie pogarsza,
3. nie zmienia predykcji,
4. dodaje wyłącznie dodatkową analizę diagnostyczną.

To jest bardzo ważny punkt do przedstawienia promotorowi.

Można to powiedzieć tak:

`main_48_cech_modelslice.py` jest narzędziem analitycznym nadbudowanym nad bazowym modelem, a nie nowym modelem predykcyjnym.


## 9. Dlaczego to porównanie jest ważne

Bez `model slicing` wiemy tylko, że model ma około `61%` accuracy na poziomie przewidywania zwycięzcy meczu.

Ale to nie wystarcza.

Bo model z wynikiem `61%` może działać na dwa sposoby:

1. może być dość równy we wszystkich typach meczów,
2. albo może działać dobrze w łatwiejszych przypadkach, a bardzo słabo w niektórych specyficznych grupach.

Slicing pozwala odróżnić te dwie sytuacje.


## 10. Najsłabsze slice’y 1D

Najważniejsze słabe grupy jednowymiarowe to:

1. `tourney_level=A` -> `50.0%`
2. `round=QF` -> `50.9%`
3. `best_of=5` -> `52.2%`
4. `handedness_matchup=L-vs-R` -> `53.6%`
5. `age_gap_bucket=3-5` -> `54.9%`
6. `tourney_level=D` -> `56.0%`

Interpretacja:

1. model ogólnie ma `61.02%`,
2. ale w tych grupach wypada wyraźnie słabiej,
3. to znaczy, że w tych kontekstach obecne cechy nie opisują dobrze sytuacji meczowej.


## 11. Najsłabsze slice’y 2D

Najbardziej problematyczne grupy dwuwymiarowe to:

1. `best_of=5 & rank_gap_bucket=51-100` -> `28.6%`
2. `tourney_level=D & form_gap_bucket=0.10-0.25` -> `29.6%`
3. `round=R128 & handedness_matchup=L-vs-R` -> `33.3%`
4. `rank_gap_bucket=0-10 & age_gap_bucket=>8` -> `33.3%`
5. `best_of=5 & form_gap_bucket=0.10-0.25` -> `33.3%`

To są szczególnie cenne informacje, bo pokazują nie tylko pojedynczy problem, ale przecięcia dwóch warunków.

Innymi słowy:

model nie tylko ma trudność z `Best of 5`, ale ma jeszcze większą trudność z `Best of 5`, gdy jednocześnie zachodzi określona konfiguracja różnicy rankingu albo różnicy formy.


## 12. Najlepsze slice’y

Analiza pokazuje też grupy, w których model działa bardzo dobrze.

Przykłady:

1. `round=R16 & rank_gap_bucket=0-10` -> `100.0%`
2. `tourney_level=M & form_gap_bucket=>0.40` -> `100.0%`
3. `round=R64 & age_gap_bucket=0-2` -> `91.7%`
4. `round=R64 & handedness_matchup=L-vs-R` -> `90.0%`

To jest ważne, bo pokazuje, że model nie jest po prostu "słaby wszędzie".

On działa dobrze w niektórych konfiguracjach, ale nierówno.


## 13. Co z tego wynika praktycznie

Najważniejszy wniosek praktyczny jest taki:

model ma sensowną średnią jakość, ale nie jest stabilny między różnymi typami meczów.

To znaczy, że:

1. średni wynik `61.02%` nie opowiada całej prawdy,
2. istnieją grupy meczów, gdzie model działa znacznie słabiej,
3. i właśnie te grupy są najlepszym kierunkiem dalszego rozwoju modelu.

Inaczej mówiąc:

`model slicing` zamienia ogólne pytanie "jak poprawić model?" na dużo bardziej konkretne pytanie:

`co trzeba dodać, żeby model lepiej działał w Best of 5, QF i L-vs-R?`


## 14. Jak można to powiedzieć promotorowi

Prosta wersja:

> Sam model z pliku `main_48_cech.py` daje około 61% skuteczności przewidywania zwycięzcy meczu.
> Dodatkowy plik `main_48_cech_modelslice.py` nie zmienia modelu, tylko analizuje, w jakich grupach meczów model działa słabiej lub lepiej.
> Dzięki temu widać, że model nie myli się losowo, tylko ma konkretne słabe obszary, na przykład mecze `Best of 5`, ćwierćfinały (`QF`) oraz mecze leworęczny kontra praworęczny (`L-vs-R`).
> To pozwala projektować kolejne cechy już nie "w ciemno", ale dokładnie pod te obszary, gdzie model najbardziej niedomaga.


## 15. Dlaczego to podejście jest wartościowe naukowo

To podejście jest wartościowe, bo nie kończy analizy na jednej średniej metryce.

W praktyce badawczej to ważne z trzech powodów:

1. daje głębsze zrozumienie zachowania modelu,
2. pozwala wskazać konkretne słabe obszary modelu,
3. uzasadnia dalszy rozwój cech w sposób metodyczny, a nie przypadkowy.

Czyli zamiast mówić:

`dodam nowe cechy, bo może pomogą`

można powiedzieć:

`dodaję nowe cechy, ponieważ slicing wykazał konkretne słabe slice’y i chcę poprawić model właśnie tam`.


## 16. Najważniejsze wnioski końcowe

### Wniosek 1

`main_48_cech_modelslice.py` nie zmienia modelu z `main_48_cech.py`.

### Wniosek 2

Oba pliki dają te same podstawowe metryki modelu:

1. `CV Accuracy = 0.6478`
2. `Validation Accuracy = 0.6331`
3. `Test Accuracy = 0.6136`
4. `Match Prediction Accuracy = 0.6102`

### Wniosek 3

`main_48_cech_modelslice.py` dodaje warstwę diagnostyczną, która pokazuje, gdzie model działa nierówno.

### Wniosek 4

Najważniejsze słabe obszary modelu to:

1. `Best of 5`,
2. `QF`,
3. `L-vs-R`,
4. niektóre przecięcia warunków, szczególnie 2D.

### Wniosek 5

Slicing daje bardzo praktyczną wartość: wskazuje, pod jakie grupy meczów warto projektować nowe cechy w kolejnych eksperymentach.


## 17. Końcowe podsumowanie jednym akapitem

Plik `main_48_cech.py` buduje i ocenia bazowy model predykcyjny dla meczów ATP. Plik `main_48_cech_modelslice.py` uruchamia ten sam model, ale dodatkowo bada jego zachowanie na konkretnych podgrupach meczów. Dzięki temu wiadomo nie tylko, że model osiąga około `61%` skuteczności, ale także gdzie dokładnie jest słabszy. W tym projekcie analiza pokazała, że szczególnie trudne są mecze `Best of 5`, ćwierćfinały (`QF`) oraz część matchupów `L-vs-R`. Oznacza to, że slicing nie jest nowym modelem, tylko narzędziem diagnostycznym, które pozwala bardziej świadomie rozwijać kolejne wersje modelu.