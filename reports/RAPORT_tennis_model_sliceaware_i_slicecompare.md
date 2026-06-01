# Raport wyjaśniający: `tennis_model_sliceaware.py` i `tennis_model_slicecompare.py`

## 1. Cel raportu

Ten dokument ma w prosty, ale dokładny sposób wyjaśnić:

1. po co w projekcie pojawił się plik `tennis_model_sliceaware.py`,
2. po co powstał plik `tennis_model_slicecompare.py`,
3. czym te pliki różnią się od `tennis_model.py` i `tennis_model_modelslice.py`,
4. za co odpowiadają najważniejsze części kodu,
5. jakie były wyniki i co z nich wynika praktycznie.

Ten raport jest rozwinięciem poprzedniego raportu o `model slicing`.


## 2. Najkrótsza odpowiedź

Najprościej można to ująć tak:

1. `tennis_model.py` to model bazowy.
2. `tennis_model_modelslice.py` nie zmienia modelu, tylko pokazuje, gdzie model działa słabo.
3. `tennis_model_sliceaware.py` to pierwsza próba poprawienia modelu dokładnie w tych słabych miejscach.
4. `tennis_model_slicecompare.py` to narzędzie porównawcze, które sprawdza, która wersja modelu wygrywa na konkretnych slice'ach.

Czyli:

1. `modelslice` odpowiada na pytanie: `gdzie model ma problem?`
2. `sliceaware` odpowiada na pytanie: `czy da się ten problem ograniczyć nowymi cechami?`
3. `slicecompare` odpowiada na pytanie: `która wersja modelu radzi sobie najlepiej i w jakich grupach meczów?`


## 3. Jak te pliki układają się w jedną całość

W tym projekcie logika rozwoju modelu wygląda tak:

1. najpierw jest model bazowy z `tennis_model.py`,
2. potem `tennis_model_modelslice.py` wykrywa słabe grupy meczów,
3. potem `tennis_model_sliceaware.py` dodaje nowe cechy specjalnie pod te słabe grupy,
4. na końcu `tennis_model_slicecompare.py` porównuje różne warianty i pokazuje, czy faktycznie coś się poprawiło.

To jest ważne, bo pokazuje metodyczny proces pracy:

1. nie dodajesz cech na ślepo,
2. tylko najpierw diagnozujesz problem,
3. potem tworzysz wariant naprawczy,
4. a potem porównujesz efekty na tych samych grupach meczów.


## 4. Co robi `tennis_model_sliceaware.py`

Ten plik jest eksperymentalnym wariantem modelu bazowego.

Jego idea jest bardzo prosta:

1. uruchomić tę samą bazową obróbkę danych co w `tennis_model.py`,
2. zostawić ten sam model `RandomForestClassifier`,
3. zostawić te same najlepsze hiperparametry znalezione wcześniej,
4. ale dodać nowe cechy kontekstowe tam, gdzie slicing wykazał problem,
5. a potem sprawdzić, czy model działa lepiej.

Czyli `sliceaware` nie jest całkowicie nowym podejściem do modelowania.

To jest raczej:

`ten sam szkielet modelu + dodatkowe cechy zaprojektowane pod słabe slice'y`.


## 5. Jakie słabe slice'y próbował naprawić `sliceaware`

W tym pliku cechy były projektowane głównie pod trzy problemy wykryte wcześniej:

1. `Best of 5`,
2. `QF`,
3. `L-vs-R`.

To są dokładnie te obszary, gdzie model bazowy wypadał słabiej niż jego średni wynik ogólny.


## 6. Co zostało zmienione względem `tennis_model.py`

Najważniejsze zmiany są takie:

1. model nadal bazuje na wynikach `tennis_model.py`, bo uruchamia go przez `runpy`,
2. stary zestaw 40 cech zostaje zachowany,
3. do tego dochodzą nowe cechy targetowane pod słabe slice'y,
4. liczba cech rośnie z `40` do `73`,
5. algorytm i hiperparametry pozostają te same, więc zmienia się głównie informacja wejściowa, a nie sam mechanizm uczenia.

To jest bardzo ważne interpretacyjnie.

Jeżeli wynik się zmienia, to główny powód jest taki, że model dostał lepszy opis sytuacji meczowej, a nie dlatego, że podmieniono cały algorytm.


## 7. Za co odpowiadają najważniejsze części `tennis_model_sliceaware.py`

### 7.1. `execute_base_pipeline_quietly()`

Ta funkcja uruchamia `tennis_model.py` w tle i pobiera z niego gotowe elementy pipeline'u.

To znaczy, że `sliceaware` nie kopiuje ręcznie całego kodu bazowego, tylko wykorzystuje już istniejący pipeline.


### 7.2. `calculate_context_form()`

Ta funkcja liczy formę gracza, ale nie ogólnie, tylko w konkretnym kontekście.

Na przykład można policzyć:

1. formę w meczach `Best of 5`,
2. formę w rundzie `QF`,
3. formę przeciwko graczom o danej ręczności,
4. formę na konkretnej nawierzchni i jednocześnie w konkretnym kontekście.

To jest ważne, bo zwykła forma ogólna może nie wystarczać.


### 7.3. `calculate_context_experience()`

Ta funkcja nie pyta: `czy gracz ostatnio wygrywał?`

Ona pyta raczej:

`ile doświadczenia ma gracz w takim typie sytuacji?`

Przykład:

1. ilu meczów `Best of 5` już doświadczył,
2. ile razy grał w `QF`,
3. ile razy był w późnych rundach.


### 7.4. `calculate_context_balance()`

Ta funkcja tworzy prosty bilans przeciwko określonej ręczności przeciwnika.

Czyli nie tylko: `czy gracz jest mocny?`, ale też:

`czy ten gracz historycznie dobrze radzi sobie z takim matchupem ręki przeciwnika?`


### 7.5. `add_targeted_slice_features()`

To jest główna funkcja budująca nowe cechy jeszcze na surowych danych meczowych.

Tutaj dla zwycięzcy i przegranego obliczane są między innymi:

1. forma w `Best of 5`,
2. doświadczenie w `Best of 5`,
3. forma w późnych rundach,
4. doświadczenie w późnych rundach,
5. forma w `QF`,
6. doświadczenie w `QF`,
7. forma w `QF` na tej samej nawierzchni,
8. forma przeciwko ręczności przeciwnika,
9. forma przeciwko ręczności przeciwnika na tej samej nawierzchni,
10. bilans przeciwko ręczności przeciwnika.


### 7.6. `attach_targeted_features()`

Po symetryzacji danych trzeba te nowe cechy poprawnie przepisać do perspektywy `p1` i `p2`.

Ta funkcja właśnie to robi.

Dodatkowo tworzy cechy różnicowe, czyli na przykład:

1. `best_of5_form_diff`,
2. `late_round_form_diff`,
3. `qf_form_diff`,
4. `qf_surface_form_diff`,
5. `opp_hand_form_diff`,
6. `opp_hand_surface_form_diff`,
7. `opp_hand_balance_diff`.

To są często najważniejsze cechy, bo model zwykle lepiej działa na różnicy między graczami niż na pojedynczej wartości jednego gracza.


### 7.7. `run_sliceaware_model()`

To jest główna funkcja wykonawcza.

Ona:

1. pobiera bazowy pipeline,
2. tworzy nowe cechy dla treningu, walidacji i testu,
3. trenuje nowy model,
4. liczy metryki,
5. porównuje je z baseline,
6. wypisuje ważność nowych cech.


## 8. Jakie dokładnie nowe cechy dodaje `sliceaware`

Najprościej można je podzielić na trzy grupy.

### 8.1. Cechy sygnalizujące kontekst meczu

Na przykład:

1. `is_best_of5`,
2. `is_qf`,
3. `is_lefty_matchup`.

To są proste znaczniki mówiące modelowi, że dany mecz należy do trudnego typu przypadku.


### 8.2. Cechy formy i doświadczenia w trudnych sytuacjach

Na przykład:

1. `p1_best_of5_form`, `p2_best_of5_form`,
2. `p1_best_of5_experience`, `p2_best_of5_experience`,
3. `p1_qf_form`, `p2_qf_form`,
4. `p1_qf_experience`, `p2_qf_experience`,
5. `p1_late_round_form`, `p2_late_round_form`.


### 8.3. Cechy matchupowe pod ręczność i nawierzchnię

Na przykład:

1. `p1_vs_opp_hand_form`, `p2_vs_opp_hand_form`,
2. `p1_vs_opp_hand_surface_form`, `p2_vs_opp_hand_surface_form`,
3. `p1_vs_opp_hand_balance`, `p2_vs_opp_hand_balance`.

To właśnie te cechy próbują uchwycić problem `L-vs-R` bardziej szczegółowo niż zwykła informacja `czy gracz jest leworęczny`.


## 9. Aktualne wyniki `tennis_model_sliceaware.py`

Po uruchomieniu aktualnej wersji tego pliku otrzymano:

1. `Validation Accuracy`: `0.6407`
2. `Test Accuracy`: `0.6127`
3. `Match Prediction Accuracy`: `0.6085`

Porównanie z baseline wygląda tak:

1. walidacja: `0.6331 -> 0.6407`, czyli `+0.0076`,
2. test: `0.6136 -> 0.6127`, czyli `-0.0008`,
3. match-level: `0.6102 -> 0.6085`, czyli `-0.0017`.


## 10. Co oznaczają te wyniki praktycznie

To oznacza, że:

1. pomysł na nowe cechy ma sens, bo walidacja wzrosła,
2. ale poprawa nie utrzymała się w sposób stabilny na końcowym teście i na poziomie rzeczywistych zwycięzców meczów,
3. więc sama wersja `sliceaware v2` jest ciekawym krokiem rozwojowym, ale nie jest jeszcze ostatecznie lepsza od baseline'u.

To bardzo ważny wniosek badawczy.

On nie oznacza, że eksperyment był zły.

On oznacza raczej, że:

1. diagnoza slice'ów była trafna,
2. część nowych sygnałów rzeczywiście wnosi informację,
3. ale obecny zestaw cech nie rozwiązuje jeszcze problemu na tyle dobrze, by dać stabilny zysk globalny.


## 11. Najważniejsze nowe cechy według ważności modelu

W aktualnym uruchomieniu najwyżej ocenione nowe cechy to między innymi:

1. `opp_hand_balance_diff`,
2. `opp_hand_form_diff`,
3. `best_of5_form_diff`,
4. `opp_hand_surface_form_diff`,
5. `late_round_form_diff`,
6. `qf_surface_form_diff`.

To daje praktyczny wniosek:

modelowi rzeczywiście przydaje się bardziej szczegółowa informacja o matchupie ręki przeciwnika, formie w trudnych kontekstach oraz specyfice późnych rund.


## 12. Co robi `tennis_model_slicecompare.py`

Ten plik nie tworzy nowego modelu.

On jest narzędziem porównawczym.

Jego zadanie polega na tym, żeby:

1. uruchomić kilka wersji modelu,
2. policzyć dla każdej z nich te same slice'y,
3. złączyć wszystko do jednej tabeli,
4. sprawdzić, która wersja jest najlepsza w danym slice'ie,
5. zapisać wynik do pliku CSV.

Czyli:

1. `sliceaware` próbuje coś poprawić,
2. `slicecompare` sprawdza, czy i gdzie faktycznie to się udało.


## 13. Jakie modele porównuje obecnie `slicecompare`

W aktualnej konfiguracji plik porównuje trzy modele:

1. `baseline` -> `tennis_model.py`,
2. `qfserve_v3` -> `tennis_model_sliceaware_qfserve_v3.py`,
3. `bestof5_v1` -> `tennis_model_sliceaware_bestof5_v1.py`.

To jest ważna uwaga:

obecna wersja `tennis_model_slicecompare.py` nie porównuje bezpośrednio pliku `tennis_model_sliceaware.py`.

Porównuje jego późniejsze eksperymentalne rozwinięcia, czyli warianty bardziej wyspecjalizowane.


## 14. Za co odpowiadają najważniejsze części `tennis_model_slicecompare.py`

### 14.1. `MODELS`

To jest słownik mówiący, jakie pliki mają być porównywane.

Jeżeli chcesz dodać lub usunąć wariant modelu z porównania, to właśnie tutaj robi się najważniejszą zmianę.


### 14.2. `SLICE_COLUMNS`

To lista atrybutów, po których liczone są slice'y.

Są tu między innymi:

1. `surface`,
2. `tourney_level`,
3. `best_of`,
4. `round`,
5. `handedness_matchup`,
6. `rank_gap_bucket`,
7. `age_gap_bucket`,
8. `form_gap_bucket`.


### 14.3. `TARGET_SLICE_PATTERNS`

To lista najważniejszych grup, na których skupia się analiza:

1. `best_of=5`,
2. `round=QF`,
3. `handedness_matchup=L-vs-R`.

To znaczy, że w kodzie od razu wiadomo, które slice'y są traktowane jako strategiczne cele poprawy.


### 14.4. `collect_slice_results()`

Ta funkcja uruchamia dany model, buduje tabelę meczów na poziomie realnych spotkań i liczy slice'y tak samo dla każdej wersji modelu.


### 14.5. `build_model_view()`

Ta funkcja bierze wynik jednego modelu i przygotowuje go do połączenia z innymi modelami.

Czyli zamienia tabelę na taki format, żeby dało się potem zestawić obok siebie accuracy różnych wariantów dla dokładnie tego samego slice'a.


### 14.6. `build_comparison_table()`

To najważniejsza część całego pliku.

Ta funkcja:

1. uruchamia wszystkie modele z listy `MODELS`,
2. pilnuje, żeby slice'y były zgodne między modelami,
3. liczy delty względem baseline,
4. wskazuje najlepszy model dla każdego slice'a,
5. porządkuje wynik końcowy.


### 14.7. `print_target_slice_summary()`

Ta funkcja wypisuje najważniejsze podsumowanie do konsoli.

Czyli użytkownik od razu widzi:

1. wynik ogólny modeli,
2. najważniejsze target slice'y,
3. gdzie dany wariant zyskał,
4. gdzie stracił.


### 14.8. `OUTPUT_CSV`

To ścieżka do głównego wyniku porównania:

`slice_comparison_all_variants.csv`

To właśnie ten plik jest końcowym zestawieniem do analizy.


## 15. Jak czytać plik `slice_comparison_all_variants.csv`

Najprościej:

1. jeden wiersz = jeden slice,
2. `support` = ile meczów należy do tego slice'a,
3. `baseline_accuracy` = accuracy modelu bazowego na tej grupie,
4. `qfserve_v3_accuracy` i `bestof5_v1_accuracy` = accuracy innych wariantów na tej samej grupie,
5. `*_accuracy_delta_vs_baseline` = o ile dany wariant jest lepszy lub gorszy od baseline'u,
6. `best_model_by_accuracy` = który model wygrał na tym konkretnym slice'ie.

To jest bardzo praktyczne, bo zamiast patrzeć tylko na jeden średni wynik, można patrzeć na konkretne typy meczów.


## 16. Najważniejsze wyniki z aktualnego `slicecompare`

Na poziomie ogólnym obecne wyniki wyglądają tak:

1. `baseline` -> około `0.6102`,
2. `qfserve_v3` -> około `0.6271`,
3. `bestof5_v1` -> około `0.6085`.

To oznacza, że globalnie najlepszym z porównywanych wariantów jest teraz `qfserve_v3`.


## 17. Co `slicecompare` pokazuje dla najważniejszych słabych grup

### 17.1. `L-vs-R`

Tutaj widać realną poprawę.

Dla slice'a `handedness_matchup=L-vs-R`:

1. `baseline` ma `53.6%`,
2. `qfserve_v3` ma `56.3%`,
3. `bestof5_v1` ma `55.0%`.

To znaczy, że kierunek z cechami matchupowymi miał sens.


### 17.2. `QF`

Tutaj poprawa nie jest przekonująca.

Dla slice'a `round=QF`:

1. `baseline` ma `50.9%`,
2. `qfserve_v3` ma również `50.9%`,
3. `bestof5_v1` spada do `45.5%`.

To znaczy, że `QF` nadal pozostaje trudnym i nierozwiązanym obszarem.


### 17.3. `Best of 5`

Tutaj także problem nie został jeszcze dobrze rozwiązany.

Dla slice'a `best_of=5`:

1. `baseline` ma `52.2%`,
2. `qfserve_v3` spada do `43.5%`,
3. `bestof5_v1` wraca do `52.2%`.

To oznacza, że nawet wariant specjalnie projektowany pod ten problem nie dał jeszcze wyraźnego przełomu.


## 18. Dlaczego `slicecompare` jest ważny badawczo

Bez tego pliku można łatwo dojść do złego wniosku.

Na przykład:

1. model może poprawić średni wynik globalny,
2. ale jednocześnie pogorszyć się dokładnie tam, gdzie najbardziej chciałeś go poprawić.

I właśnie to jest jedna z najważniejszych rzeczy, które pokazuje `slicecompare`.

On uczy, że:

1. `lepszy wynik ogólny` nie znaczy automatycznie `lepszy model w każdym ważnym przypadku`,
2. trzeba patrzeć także na to, jakie grupy zostały poprawione, a jakie popsute.


## 19. Najprostsza różnica między `sliceaware` a `slicecompare`

Najprościej:

1. `sliceaware` coś zmienia w modelu,
2. `slicecompare` niczego nie zmienia w modelu, tylko porównuje wyniki kilku wersji.

Jeszcze prościej:

1. `sliceaware` = eksperyment naprawczy,
2. `slicecompare` = tablica wyników i dowód, czy eksperyment się opłacił.


## 20. Co można powiedzieć promotorowi

Prosta wersja:

> Po wykryciu słabych grup meczów za pomocą model slicing przygotowano wariant `sliceaware`, który dodaje cechy kontekstowe specjalnie pod te grupy, między innymi pod mecze Best of 5, ćwierćfinały oraz matchupy leworęczny kontra praworęczny. Następnie użyto pliku `slicecompare`, aby porównać różne warianty modeli na dokładnie tych samych slice'ach. Dzięki temu można sprawdzić nie tylko, czy model poprawił średni wynik ogólny, ale też czy faktycznie poprawił się tam, gdzie wcześniej był najsłabszy.


## 21. Najważniejsze wnioski końcowe

### Wniosek 1

`tennis_model_sliceaware.py` jest pierwszym eksperymentem, który próbuje poprawić model dokładnie tam, gdzie wcześniejszy slicing pokazał problemy.


### Wniosek 2

Największa zmiana w `sliceaware` nie dotyczy algorytmu, tylko dodatkowych cech kontekstowych.


### Wniosek 3

Aktualna wersja `sliceaware` poprawia walidację, ale nie daje jeszcze stabilnej poprawy na końcowym teście i na poziomie realnych zwycięzców meczów.


### Wniosek 4

`tennis_model_slicecompare.py` jest narzędziem, które pozwala uczciwie porównywać różne wersje modelu na tych samych slice'ach.


### Wniosek 5

Aktualne porównania pokazują, że najłatwiej poprawić część slice'ów `L-vs-R`, natomiast `QF` i `Best of 5` nadal pozostają trudnymi obszarami.


## 22. Końcowe podsumowanie jednym akapitem

Plik `tennis_model_sliceaware.py` to eksperymentalna wersja modelu, która wykorzystuje wyniki wcześniejszego model slicing i dodaje nowe cechy specjalnie pod słabe grupy meczów, takie jak `Best of 5`, `QF` i `L-vs-R`. Nie zmienia on samego typu modelu, ale wzbogaca opis sytuacji meczowej. Z kolei `tennis_model_slicecompare.py` nie buduje nowego modelu, tylko porównuje kilka wariantów na dokładnie tych samych slice'ach i zapisuje wynik do wspólnego CSV. Dzięki temu w projekcie widać cały logiczny ciąg pracy: najpierw diagnoza problemu, potem próba jego naprawy, a na końcu twarde porównanie, czy poprawa rzeczywiście zaszła i w których grupach meczów.