# Podsumowanie wariantów slice-aware i porównania modeli

## 1. Najkrótszy wniosek

Najważniejszy wniosek z całego eksperymentu jest taki:

1. sam `model slicing` dobrze wskazał, gdzie model ma problemy,
2. najlepszym globalnie wariantem okazał się `tennis_model_sliceaware_qfserve_v3.py`,
3. największe i najbardziej powtarzalne poprawy pojawiły się w slice'ach typu `L-vs-R`,
4. slice'y `QF` i `Best of 5` nadal nie zostały rozwiązane w pełni.


## 2. Logika całego flow

Cały przepływ pracy wyglądał tak:

1. `tennis_model.py` buduje model bazowy,
2. `tennis_model_modelslice.py` pokazuje, gdzie ten model działa słabiej,
3. `tennis_model_sliceaware.py` robi pierwszą próbę poprawy tych słabych miejsc,
4. `tennis_model_sliceaware_qfserve_v3.py` robi bardziej dopracowaną próbę poprawy QF i kontekstu serwisowego,
5. `tennis_model_sliceaware_bestof5_v1.py` robi specjalny wariant pod mecze `Best of 5`,
6. `tennis_model_slicecompare.py` porównuje wszystkie warianty na tych samych slice'ach.

To znaczy, że projekt rozwija się metodycznie:

1. najpierw diagnoza,
2. potem poprawka,
3. potem porównanie, czy poprawka rzeczywiście zadziałała.


## 3. Co robią slice'y w tych modelach

W tym projekcie `slice` to po prostu konkretna grupa meczów, na przykład:

1. `round=QF`,
2. `best_of=5`,
3. `handedness_matchup=L-vs-R`.

Sens tego podejścia jest prosty:

1. model może mieć sensowny wynik ogólny,
2. ale jednocześnie bardzo źle działać w kilku konkretnych typach meczów,
3. więc zamiast poprawiać model „wszędzie naraz”, próbujemy poprawiać go dokładnie tam, gdzie rzeczywiście się psuje.

Właśnie dlatego warianty `sliceaware` dodają cechy kontekstowe pod konkretne slice'y.


## 4. Krótkie role poszczególnych modeli

### `tennis_model.py`

To jest punkt odniesienia, czyli model bazowy.


### `tennis_model_sliceaware.py`

To jest pierwszy ogólny wariant naprawczy. Dodaje cechy pod `Best of 5`, `QF` i `L-vs-R`, ale jeszcze dość szeroko i eksperymentalnie.


### `tennis_model_sliceaware_qfserve_v3.py`

To jest najmocniejszy wariant globalny. Łączy lepszy kontekst `QF` z bardziej szczegółowymi cechami serwisu i returnu.


### `tennis_model_sliceaware_bestof5_v1.py`

To jest wariant specjalizowany pod `Best of 5`. Skupia się na wytrzymałości, długich meczach i jakości gry w dłuższym formacie.


### `tennis_model_slicecompare.py`

To jest narzędzie porównawcze. Ono nie poprawia modelu, tylko pokazuje, która wersja modelu wygrała na danym slice'ie.


## 5. Wyniki globalne modeli

Na poziomie `match-level accuracy` wyniki wyglądają tak:

1. `baseline` -> `0.6102`
2. `sliceaware` -> `0.6085`
3. `qfserve_v3` -> `0.6271`
4. `bestof5_v1` -> `0.6085`

Wniosek globalny jest prosty:

1. najlepszy ogólnie jest `qfserve_v3`,
2. zwykły `sliceaware` i `bestof5_v1` nie przebiły baseline'u globalnie,
3. więc nie każda próba poprawy slice'ów daje poprawę całego modelu.


## 6. Co naprawdę się udało

Najbardziej udał się kierunek związany z matchupami `L-vs-R`.

Najważniejsze przykłady:

1. ogólny slice `handedness_matchup=L-vs-R` poprawił się z `53.6%` do `55.6%` w `sliceaware` i do `56.3%` w `qfserve_v3`,
2. `qfserve_v3` bardzo dobrze poprawiał wiele przecięć `L-vs-R`,
3. `sliceaware` też poprawił część slice'ów `L-vs-R`, ale mniej konsekwentnie,
4. `bestof5_v1` miał lokalne sukcesy, ale nie zmienił obrazu całości tak dobrze jak `qfserve_v3`.

Czyli najważniejszy praktyczny sukces to:

`lepsze modelowanie matchupów ręczności i kontekstu gry przeciwnika`.


## 7. Który slice był najlepszy

Tu trzeba rozróżnić dwa znaczenia słowa „najlepszy”.

### 7.1. Największy zysk procentowy na target slice

Największy dodatni skok w target slice'ach miał `qfserve_v3` dla:

1. `round=R128 & handedness_matchup=L-vs-R`
2. wzrost z `33.3%` do `66.7%`
3. czyli `+33.3 p.p.`

To jest bardzo mocny wzrost, ale trzeba pamiętać, że support tego slice'a wynosi tylko `9`, więc to jest sygnał ciekawy, ale na małej grupie.


### 7.2. Najbardziej wiarygodny praktycznie slice z dobrym supportem

Jeżeli patrzeć na slice'y z bardziej sensowną liczbą meczów, to bardzo mocny wynik dał `qfserve_v3` dla:

1. `tourney_level=M & handedness_matchup=L-vs-R`
2. wzrost z `61.8%` do `70.6%`
3. czyli `+8.8 p.p.` przy `34` meczach

To jest bardziej przekonujące badawczo niż skok na bardzo małej próbce.


### 7.3. Najbardziej użyteczny szeroki slice

Najbardziej praktyczny szeroki slice to po prostu:

1. `handedness_matchup=L-vs-R`
2. baseline: `53.6%`
3. `sliceaware`: `55.6%`
4. `qfserve_v3`: `56.3%`
5. `bestof5_v1`: `55.0%`

To jest ważne, bo tutaj support wynosi aż `151` meczów, więc poprawa nie dotyczy pojedynczej niszy, tylko większej grupy danych.


## 8. Co się nie udało

Dwie rzeczy nadal są słabe.

### 8.1. `QF`

Dla głównego slice'a `round=QF`:

1. baseline ma `50.9%`,
2. `sliceaware` spada do `45.5%`,
3. `qfserve_v3` zostaje na `50.9%`,
4. `bestof5_v1` też spada do `45.5%`.

To znaczy, że `QF` nadal jest trudnym obszarem i żaden wariant nie dał tu wyraźnego przełomu.


### 8.2. `Best of 5`

Dla głównego slice'a `best_of=5`:

1. baseline ma `52.2%`,
2. `sliceaware` ma `52.2%`,
3. `qfserve_v3` spada do `43.5%`,
4. `bestof5_v1` wraca tylko do `52.2%`.

To jest bardzo ważny wniosek:

wariant projektowany specjalnie pod `Best of 5` nie poprawił tego slice'a ponad baseline.


## 9. Który model wygrał najczęściej na target slice'ach

Po zliczeniu najlepszego modelu dla target slice'ów wyszło:

1. `baseline` wygrywa w `36` target slice'ach,
2. `qfserve_v3` wygrywa w `17`,
3. `sliceaware` wygrywa w `6`,
4. `bestof5_v1` wygrywa w `4`.

To oznacza dwie rzeczy jednocześnie:

1. baseline nadal jest zaskakująco trudny do pobicia w wielu grupach,
2. ale gdy już jakaś poprawa rzeczywiście działa, to najczęściej najlepiej robi to `qfserve_v3`.


## 10. Najważniejsza interpretacja badawcza

Najważniejsza lekcja z tych eksperymentów jest taka:

1. `model slicing` naprawdę pomaga znaleźć słabe miejsca modelu,
2. ale poprawienie jednego słabego slice'a nie gwarantuje poprawy całego modelu,
3. najlepsze efekty dają cechy, które opisują kontekst meczu bardziej szczegółowo,
4. szczególnie dobrze widać to przy matchupach `L-vs-R` i cechach serwisowo-kontekstowych,
5. `QF` i `Best of 5` wciąż wymagają dalszej pracy.


## 11. Końcowy wniosek

Jeżeli spojrzeć na cały projekt jako na serię eksperymentów, to wynik jest spójny. `tennis_model_modelslice.py` dobrze pokazał, gdzie model ma problemy. `tennis_model_sliceaware.py` był sensowną pierwszą próbą poprawy tych obszarów, ale jeszcze bez stabilnego zysku globalnego. Najlepszym rozwinięciem okazał się `tennis_model_sliceaware_qfserve_v3.py`, który poprawił wynik ogólny i wiele slice'ów związanych z `L-vs-R`. Jednocześnie `QF` i `Best of 5` nadal pozostają trudnymi obszarami, więc dalsze prace powinny iść w stronę jeszcze lepszych cech kontekstowych właśnie dla tych dwóch typów meczów.