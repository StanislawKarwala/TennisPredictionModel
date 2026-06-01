# Co zrobiłem przy modelu predykcji tenisa — opis dla promotora

Mój celem było sprawdzić, czy da się poprawić skuteczność modelu, który przewiduje
zwycięzcę meczu tenisowego (ATP), i — co ważniejsze — rzetelnie odróżnić to, co
naprawdę pomaga, od tego, co tylko wygląda, że pomaga.

Zacząłem od **naprawienia sposobu mierzenia skuteczności**. Okazało się, że
dotychczasowa metryka liczyła trafność jednostronnie i mogła zawyżać wynik.
Poprawiłem ją tak, żeby uśredniała obie perspektywy meczu — od tego momentu mam
pewność, że liczby, które raportuję, są wiarygodne. Naprawiłem też kilka innych
drobnych błędów (m.in. cechę, która liczyła się na całej karierze gracza zamiast
na bieżącym turnieju) i zmieniłem kryterium doboru ustawień modelu na takie, które
lepiej pasuje do zadania prawdopodobieństwowego.

Następnie **sprawdziłem, czy inny algorytm pomoże**. Zamiast lasu losowego
(Random Forest) przetestowałem gradient boosting — XGBoost i HistGradientBoosting.
Nie pobiły lasu losowego ani na małych, ani później na dużych danych.

Potem **dodałem nowe cechy**, o których teoria mówi, że powinny pomóc: prędkość
kortu (na szybkim korcie mocny serwis daje większą przewagę), zmęczenie gracza
(dni odpoczynku, minuty zagrane w turnieju) oraz rating Elo dostosowany do
nawierzchni (system oceny siły gracza znany z szachów). Na pojedynczym sezonie
te cechy wyglądały świetnie — poprawa nawet o 2 punkty procentowe.

Testowałem też **warianty modelu wyspecjalizowane pod konkretne typy meczów**
(slice-aware: `bestof5` dla meczów Grand Slamów do trzech wygranych setów oraz
`qfserve` skupiony na ćwierćfinałach i jakości serwisu). Na pojedynczym sezonie
wyglądały lepiej od modelu głównego, ale **na większej ilości danych okazały się
od niego słabsze** — czyli ich przewaga też była przypadkiem.

I tu doszedłem do **najważniejszego wniosku**. Sprawdziłem te cechy i warianty
uczciwie, trenując model na starszych sezonach i testując na kolejnych, przez kilka
lat z rzędu (tzw. walidacja walk-forward) z testem istotności statystycznej. Okazało
się, że ta poprawa **była przypadkiem** — przez kilka sezonów zysk znikał i nic nie
poprawiało modelu w sposób istotny statystycznie. To pokazało mi, że na małych
danych pojedynczy test potrafi „skłamać", a dopiero wieloletnia walidacja mówi prawdę.

Na koniec **dodałem znacznie więcej danych** — cały tenis ATP od 2000 roku, czyli
ponad 70 tysięcy meczów (kilkadziesiąt razy więcej niż na początku) — i powtórzyłem
porównanie algorytmów. Wynik był ten sam: niezależnie od algorytmu i ilości danych
model osiągał około **65% trafności**.

**Mój główny wniosek:** skuteczność na poziomie ~65% to naturalny sufit dla tego
typu danych (ranking, forma, statystyki serwisu) — co potwierdza też literatura
naukowa. Nie da się go przebić ani lepszym algorytmem, ani większą ilością danych,
ani kolejnymi cechami tego rodzaju. Co istotne, mój model bije „naiwne" zgadywanie
(„wygrywa gracz wyżej w rankingu") tylko o około 1 punkt procentowy — większość
trafności bierze się po prostu stąd, że faworyci zwykle wygrywają. Jedyne, co
realnie podniosłoby wynik powyżej tego sufitu, to kursy bukmacherskie (świadomie
ich nie używam) albo zaawansowane systemy ratingowe.

Dlatego najważniejszą wartością tej pracy nie jest sama liczba trafności, lecz
**rygor metodologiczny**: pokazałem, że model trzeba walidować na wielu sezonach,
bo pojedynczy test wprowadza w błąd, i że dla tych danych ~65% jest realnym,
odpornym sufitem. To uczciwy, dobrze udokumentowany wynik.
