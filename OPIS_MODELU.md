# Model predykcji wyników meczów tenisowych (ATP)

## Spis treści

1. [Przegląd modelu](#1-przegląd-modelu)
2. [Dane źródłowe — kolumny pliku CSV](#2-dane-źródłowe--kolumny-pliku-csv)
3. [Kolumny używane w modelu](#3-kolumny-używane-w-modelu)
4. [Kolumny nieużywane](#4-kolumny-nieużywane)
5. [Pełna lista cech modelu (40)](#5-pełna-lista-cech-modelu-40)
6. [Logika działania — krok po kroku](#6-logika-działania--krok-po-kroku)
7. [Techniki i „sztuczki&#34; poprawiające model](#7-techniki-i-sztuczki-poprawiające-model)
8. [Wyniki modelu](#8-wyniki-modelu)
9. [Ważność cech (Feature Importance)](#9-ważność-cech-feature-importance)
10. [Ograniczenia i uwagi](#10-ograniczenia-i-uwagi)

---

## 1. Przegląd modelu

| Parametr                      | Wartość                                                     |
| ----------------------------- | ------------------------------------------------------------- |
| **Algorytm**            | Random Forest Classifier                                      |
| **Optymalizacja**       | RandomizedSearchCV (50 iteracji)                              |
| **Walidacja krzyżowa** | TimeSeriesSplit (5 foldów, chronologiczny)                   |
| **Podział danych**     | Chronologiczny: 60% trening / 20% walidacja / 20% test        |
| **Dane treningowe**     | Sezon ATP 2024 (2950 meczów po filtracji)                    |
| **Dane historyczne**    | Sezony ATP 2018–2023 (14 945 meczów) — redukcja cold-start |
| **Liczba cech**         | 40                                                            |
| **Symetryzacja**        | Tak — każdy mecz → 2 przykłady treningowe                 |
| **Ziarno losowości**   | `RANDOM_STATE = 42`                                         |

**Cel modelu:** Dla danej pary graczy i kontekstu meczu (nawierzchnia, turniej, runda) przewidzieć, który z dwóch graczy wygra mecz.

---

## 2. Dane źródłowe — kolumny pliku CSV

Dane pochodzą z repozytorium **Jeff Sackmann / tennis_atp**. Każdy wiersz opisuje jeden mecz z perspektywy zwycięzcy (`winner_*`) i przegranego (`loser_*`). Plik CSV zawiera **50 kolumn**:

| #  | Kolumna                | Typ   | Opis                                                                                                                                             | Używana? |
| -- | ---------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------ | :-------: |
| 1  | `tourney_id`         | str   | Unikalny identyfikator turnieju (np.`2024-580`)                                                                                                |    ❌    |
| 2  | `tourney_name`       | str   | Nazwa turnieju (np.`Australian Open`)                                                                                                          |    ❌    |
| 3  | `surface`            | str   | Nawierzchnia kortu:`Hard`, `Clay`, `Grass`                                                                                                 |    ✅    |
| 4  | `draw_size`          | int   | Rozmiar drabinki turnieju (32, 64, 128)                                                                                                          |    ❌    |
| 5  | `tourney_level`      | str   | Poziom turnieju:`G` (Grand Slam), `M` (Masters 1000), `500`, `250`, `F` (Finals), `D` (Davis Cup), `A` (inne), `O` (Olimpiada)   |    ✅    |
| 6  | `indoor`             | str   | `I` = hala, `O` = otwarte korty                                                                                                              |    ❌    |
| 7  | `tourney_date`       | int   | Data rozpoczęcia turnieju (format `YYYYMMDD`)                                                                                                 |    ✅*    |
| 8  | `match_num`          | int   | Numer meczu w turnieju (porządek rozgrywania)                                                                                                   |    ✅*    |
| 9  | `winner_id`          | str   | Identyfikator zwycięzcy w bazie ATP                                                                                                             |    ❌    |
| 10 | `winner_seed`        | float | Rozstawienie zwycięzcy (NaN jeśli nierozstawiony)                                                                                              |    ❌    |
| 11 | `winner_entry`       | str   | Sposób wejścia do turnieju:`WC` (wild card), `Q` (kwalifikacje), `LL` (lucky loser), `PR` (protected ranking), `SE` (special exempt) |    ❌    |
| 12 | `winner_name`        | str   | Imię i nazwisko zwycięzcy                                                                                                                      |    ✅    |
| 13 | `winner_hand`        | str   | Ręczność zwycięzcy:`R` (praworęczny), `L` (leworęczny)                                                                                 |    ✅    |
| 14 | `winner_ht`          | float | Wzrost zwycięzcy w cm                                                                                                                           |    ✅    |
| 15 | `winner_ioc`         | str   | Kraj zwycięzcy (kod 3-literowy, np.`ESP`)                                                                                                     |    ❌    |
| 16 | `winner_age`         | float | Wiek zwycięzcy w dniu meczu (z ułamkiem roku)                                                                                                  |    ✅    |
| 17 | `winner_rank`        | float | Ranking ATP zwycięzcy w dniu turnieju                                                                                                           |    ✅    |
| 18 | `winner_rank_points` | float | Punkty rankingowe ATP zwycięzcy                                                                                                                 |    ✅    |
| 19 | `loser_id`           | str   | Identyfikator przegranego w bazie ATP                                                                                                            |    ❌    |
| 20 | `loser_seed`         | float | Rozstawienie przegranego (NaN jeśli nierozstawiony)                                                                                             |    ❌    |
| 21 | `loser_entry`        | str   | Sposób wejścia do turnieju (jak winner_entry)                                                                                                  |    ❌    |
| 22 | `loser_name`         | str   | Imię i nazwisko przegranego                                                                                                                     |    ✅    |
| 23 | `loser_hand`         | str   | Ręczność przegranego:`R` / `L`                                                                                                            |    ✅    |
| 24 | `loser_ht`           | float | Wzrost przegranego w cm                                                                                                                          |    ✅    |
| 25 | `loser_ioc`          | str   | Kraj przegranego                                                                                                                                 |    ❌    |
| 26 | `loser_age`          | float | Wiek przegranego                                                                                                                                 |    ✅    |
| 27 | `loser_rank`         | float | Ranking ATP przegranego                                                                                                                          |    ✅    |
| 28 | `loser_rank_points`  | float | Punkty rankingowe ATP przegranego                                                                                                                |    ✅    |
| 29 | `score`              | str   | Wynik meczu set po secie (np.`6-3 7-6(4)`)                                                                                                     |    ❌    |
| 30 | `best_of`            | int   | Maksymalna liczba setów:`3` lub `5`                                                                                                         |    ✅    |
| 31 | `round`              | str   | Runda turnieju:`R128`, `R64`, `R32`, `R16`, `QF`, `SF`, `F`, `RR`, `BR`                                                        |    ✅    |
| 32 | `minutes`            | float | Czas trwania meczu w minutach                                                                                                                    |    ❌    |
| 33 | `w_ace`              | float | Asy serwisowe zwycięzcy                                                                                                                         |    ✅    |
| 34 | `w_df`               | float | Podwójne błędy serwisowe zwycięzcy                                                                                                           |    ✅    |
| 35 | `w_svpt`             | float | Łączna liczba punktów serwisowych zwycięzcy                                                                                                  |    ✅    |
| 36 | `w_1stIn`            | float | Pierwsze serwisy w korcie (zwycięzca)                                                                                                           |    ✅    |
| 37 | `w_1stWon`           | float | Punkty wygrane na 1. serwisie (zwycięzca)                                                                                                       |    ✅    |
| 38 | `w_2ndWon`           | float | Punkty wygrane na 2. serwisie (zwycięzca)                                                                                                       |    ✅    |
| 39 | `w_SvGms`            | float | Gemy serwisowe rozegrane (zwycięzca)                                                                                                            |    ✅    |
| 40 | `w_bpSaved`          | float | Break pointy obronione (zwycięzca)                                                                                                              |    ✅    |
| 41 | `w_bpFaced`          | float | Break pointy zmierzone (zwycięzca)                                                                                                              |    ✅    |
| 42 | `l_ace`              | float | Asy serwisowe przegranego                                                                                                                        |    ✅    |
| 43 | `l_df`               | float | Podwójne błędy serwisowe przegranego                                                                                                          |    ✅    |
| 44 | `l_svpt`             | float | Łączna liczba punktów serwisowych przegranego                                                                                                 |    ✅    |
| 45 | `l_1stIn`            | float | Pierwsze serwisy w korcie (przegrany)                                                                                                            |    ✅    |
| 46 | `l_1stWon`           | float | Punkty wygrane na 1. serwisie (przegrany)                                                                                                        |    ✅    |
| 47 | `l_2ndWon`           | float | Punkty wygrane na 2. serwisie (przegrany)                                                                                                        |    ✅    |
| 48 | `l_SvGms`            | float | Gemy serwisowe rozegrane (przegrany)                                                                                                             |    ✅    |
| 49 | `l_bpSaved`          | float | Break pointy obronione (przegrany)                                                                                                               |    ✅    |
| 50 | `l_bpFaced`          | float | Break pointy zmierzone (przegrany)                                                                                                               |    ✅    |

> \*`tourney_date` i `match_num` nie są cechami modelu — służą wyłącznie do **sortowania chronologicznego** danych.

**Podsumowanie**: Z 50 kolumn CSV model korzysta z **34** (w tym 2 tylko do sortowania). Nie są to jednak bezpośrednie cechy — surowe kolumny przechodzą przez transformacje (log, rolling avg, kodowanie) i generują 40 finalnych cech.

---

## 3. Kolumny używane w modelu

### 3.1. Cechy kontekstowe meczu

| Kolumna CSV       | → Cecha modelu   | Transformacja                                                      | Wpływ na model                                                                                                                               |
| ----------------- | ----------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `surface`       | `surface`       | LabelEncoder (Clay=0, Grass=1, Hard=2)                             | Nawierzchnia determinuje styl gry — np. serwisanci dominują na trawie, baseliners na ziemi. Podstawowa cecha kontekstowa.                   |
| `tourney_level` | `tourney_level` | LabelEncoder (250, 500, A, D, F, G, M, O)                          | Grand Slamy (G) mają 5 setów i inną dynamikę niż turnieje 250. Model uczy się, że na większych turniejach faworyt wygrywa częściej. |
| `best_of`       | `best_of`       | Bezpośrednio (wartość: 3 lub 5)                                 | W meczach best-of-5 (Grand Slamy) niespodzianki zdarzają się rzadziej — faworyt ma więcej szans, by odwrócić niekorzystny wynik.        |
| `round`         | `round_num`     | Mapowanie ordynalne (R128=1, R64=2, R32=3, R16=4, QF=5, SF=6, F=7) | Wczesne rundy mają więcej niespodzianek. W późniejszych rundach grają silniejsi gracze, co zmienia profil predykcji.                     |

### 3.2. Cechy statyczne graczy

| Kolumna CSV                                    | → Cecha modelu                                             | Transformacja                           | Wpływ na model                                                                                                                                                                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `winner_rank` / `loser_rank`               | `p1_rank_log`, `p2_rank_log`, `rank_diff`             | Logarytm naturalny:`log(rank)`        | **Najsilniejsza grupa cech.** Ranking ATP to złoty standard siły gracza. Transformacja logarytmiczna kompresuje ogon — różnica między #1 a #10 jest ważniejsza niż między #90 a #100. `rank_diff` (różnica logarytmów) to osobno obliczana cecha różnicowa. |
| `winner_rank_points` / `loser_rank_points` | `p1_rank_pts_log`, `p2_rank_pts_log`, `rank_pts_diff` | Logarytm naturalny:`log(rank_points)` | **Najważniejsza cecha różnicowa** (importance=0.107). Daje większą granularność niż sam ranking — gracz #10 z 3000 pkt to inny gracz niż #10 z 4500 pkt.                                                                                                          |
| `winner_age` / `loser_age`                 | `p1_age`, `p2_age`, `age_diff`                        | Bezpośrednio (wiek z ułamkiem roku)   | Wiek koreluje z doświadczeniem (starsi) i atletyzmem (młodsi).`age_diff` pomaga modelowi wychwycić efekt młodość-vs-doświadczenie.                                                                                                                                       |
| `winner_ht` / `loser_ht`                   | `p1_ht`, `p2_ht`, `ht_diff`                           | Bezpośrednio (cm)                      | Wzrost wpływa na siłę serwisu (wyżsi → silniejszy serwis) i mobilność (niżsi → lepsza praca nóg).`ht_diff` jest cechą różnicową.                                                                                                                                  |
| `winner_hand` / `loser_hand`               | `p1_is_lefty`, `p2_is_lefty`                            | Binaryzacja:`L` → 1, `R` → 0      | Leworęczni stanowią ~14% graczy. Praworęczni mają mniej okazji do gry z leworęcznymi, co daje leworęcznym element zaskoczenia (nietypowe spiny, kąty).                                                                                                                     |

### 3.3. Cechy dynamiczne (obliczane z historii meczów)

Te cechy nie wynikają bezpośrednio z jednej kolumny CSV — obliczane są z historii meczów danego gracza. Korzystają z kolumn `winner_name`, `loser_name`, `surface` oraz 18 kolumn serwisowych (`w_ace`...`l_bpFaced`).

| Cecha modelu                             | Obliczanie                                                                                                                                   | Wpływ na model                                                                                                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `p1_form`, `p2_form`                 | Wskaźnik zwycięstw z ostatnich 10 meczów gracza (sliding window). Zakres [0.0, 1.0]. Domyślnie 0.5 (brak historii).                      | Wychwytuje bieżącą formę sportową. Gracz z serią 8 wygranych z rzędu jest w lepszej dyspozycji niż przy 2/10.                                             |
| `p1_surface_form`, `p2_surface_form` | Jak forma ogólna, ale liczona tylko z meczów na nawierzchni bieżącego meczu. Fallback na formę ogólną jeśli <3 mecze na nawierzchni. | Specjalizacja nawierzchniowa — Nadal ma lepszą formę na Clay niż na Hard. Pozwala modelowi uwzględnić tę różnicę.                                       |
| `p1_h2h`                               | Bilans bezpośrednich spotkań: wygrane p1 − wygrane p2 (z całej dostępnej historii).                                                     | Psychologiczna i stylistyczna przewaga w bezpośrednich pojedynkach. Niektóre pary mają wyraźną asymetrię (np. Djokovic–Nadal na różnych nawierzchniach). |
| `form_diff`                            | `p1_form − p2_form`                                                                                                                       | Cecha różnicowa — szybkie porównanie kto jest w lepszej formie.                                                                                               |

### 3.4. Statystyki serwisowe i returnowe (rolling average z 10 meczów)

Każda z poniższych 8 statystyk obliczana jest osobno dla obu graczy, dając **16 cech**. Obliczane jako średnia z ostatnich 10 meczów gracza. Dla graczy bez historii stosowane są wartości domyślne (średnie tourowe ATP).

| Cecha                 | Wzór                                             | Domyślna | Wpływ na model                                                            |
| --------------------- | ------------------------------------------------- | --------- | -------------------------------------------------------------------------- |
| `ace_rate`          | asy / punkty serwisowe                            | 0.08      | Mierzy siłę serwisu. Wysoka → duży serwis (Isner, Opelka).             |
| `df_rate`           | podwójne błędy / punkty serwisowe              | 0.03      | Mierzy niestabilność serwisu. Wysoka → ryzykowny gracz.                 |
| `first_in_pct`      | 1. serwisy In / wszystkie punkty serwisowe        | 0.60      | Procent trafionych pierwszych serwisów. Wysoka → stabilny serwis.        |
| `first_won_pct`     | punkty wygrane na 1. serwisie / 1. serwisy In     | 0.70      | **Jedna z najważniejszych cech.** Efektywność pierwszego serwisu. |
| `second_won_pct`    | punkty wygrane na 2. serwisie / (svpt − 1stIn)   | 0.50      | Efektywność drugiego serwisu. Niska → gracz podatny na ataki returnem.  |
| `bp_save_pct`       | break pointy obronione / break pointy zmierzone   | 0.60      | Zdolność obrony pod presją. Wysoka → gracz odporny mentalnie.          |
| `bp_faced_per_game` | break pointy zmierzone / gemy serwisowe           | 0.40      | Ile break pointów gracz oddaje na gem. Niska → trudny do przełamania.   |
| `return_pts_won`    | (opp_svpt − opp_1stWon − opp_2ndWon) / opp_svpt | 0.35      | Zdolność returnowa. Wysoka → gracz potrafi przełamywać rywala.        |

---

## 4. Kolumny nieużywane

| Kolumna                            | Powód pominięcia                                                                                        |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `tourney_id`, `tourney_name`   | Identyfikatory — zbyt wiele unikalnych wartości, brak generalizacji.                                    |
| `draw_size`                      | Redundantne z `tourney_level` (Grand Slam = 128, Masters = 64/96, itd.).                                |
| `indoor`                         | 8% NaN-ów. Mogłoby wnosić informacje o warunkach (hala vs outdoor), ale strata danych jest zbyt duża. |
| `winner_id` / `loser_id`       | Identyfikatory wewnętrzne bazy ATP — nie niosą informacji predykcyjnej.                                |
| `winner_seed` / `loser_seed`   | ~60% NaN (większość graczy nie jest rozstawiona). Częściowo redundantne z rankingiem.                |
| `winner_entry` / `loser_entry` | ~85% NaN (większość graczy wchodzi bezpośrednio = brak wartości).                                    |
| `winner_ioc` / `loser_ioc`     | ~80 unikalnych krajów — zbyt rzadkie kategorie. RF nie poradzi sobie z tak sparse'owym kodowaniem.      |
| `score`                          | ⚠️**Data leakage!** Wynik meczu znany dopiero po zakończeniu gry.                                |
| `minutes`                        | ⚠️**Data leakage!** Czas trwania meczu znany dopiero po zakończeniu gry.                         |

---

## 5. Pełna lista cech modelu (40)

### Cechy kontekstowe (4 cechy)

| # | Nazwa cechy       | Źródło                          |
| :-: | ----------------- | ---------------------------------- |
| 1 | `surface`       | LabelEncoded nawierzchnia          |
| 2 | `tourney_level` | LabelEncoded poziom turnieju       |
| 3 | `best_of`       | Bezpośrednio z CSV (3 lub 5)      |
| 4 | `round_num`     | Ordynalnie zakodowana runda (1–7) |

### Cechy statyczne gracza 1 (5 cech)

| # | Nazwa cechy         | Źródło                         |
| :-: | ------------------- | --------------------------------- |
| 5 | `p1_rank_log`     | log(ranking ATP)                  |
| 6 | `p1_rank_pts_log` | log(punkty rankingowe)            |
| 7 | `p1_age`          | Wiek z ułamkiem roku             |
| 8 | `p1_ht`           | Wzrost w cm                       |
| 9 | `p1_is_lefty`     | 1 = leworęczny, 0 = praworęczny |

### Cechy statyczne gracza 2 (5 cech)

| # | Nazwa cechy         | Źródło              |
| :-: | ------------------- | ---------------------- |
| 10 | `p2_rank_log`     | log(ranking ATP)       |
| 11 | `p2_rank_pts_log` | log(punkty rankingowe) |
| 12 | `p2_age`          | Wiek                   |
| 13 | `p2_ht`           | Wzrost                 |
| 14 | `p2_is_lefty`     | Ręczność            |

### Cechy dynamiczne (5 cech)

| # | Nazwa cechy         | Źródło                                |
| :-: | ------------------- | ---------------------------------------- |
| 15 | `p1_h2h`          | Bilans H2H (z perspektywy P1)            |
| 16 | `p1_form`         | Forma ogólna P1 (win rate z 10 meczów) |
| 17 | `p2_form`         | Forma ogólna P2                         |
| 18 | `p1_surface_form` | Forma P1 na nawierzchni meczu            |
| 19 | `p2_surface_form` | Forma P2 na nawierzchni meczu            |

### Statystyki serwisowe P1 (8 cech)

| # | Nazwa cechy              |
| :-: | ------------------------ |
| 20 | `p1_ace_rate`          |
| 21 | `p1_df_rate`           |
| 22 | `p1_first_in_pct`      |
| 23 | `p1_first_won_pct`     |
| 24 | `p1_second_won_pct`    |
| 25 | `p1_bp_save_pct`       |
| 26 | `p1_bp_faced_per_game` |
| 27 | `p1_return_pts_won`    |

### Statystyki serwisowe P2 (8 cech)

| # | Nazwa cechy              |
| :-: | ------------------------ |
| 28 | `p2_ace_rate`          |
| 29 | `p2_df_rate`           |
| 30 | `p2_first_in_pct`      |
| 31 | `p2_first_won_pct`     |
| 32 | `p2_second_won_pct`    |
| 33 | `p2_bp_save_pct`       |
| 34 | `p2_bp_faced_per_game` |
| 35 | `p2_return_pts_won`    |

### Cechy różnicowe (5 cech)

| # | Nazwa cechy       | Wzór                                |
| :-: | ----------------- | ------------------------------------ |
| 36 | `rank_diff`     | log(rank_p1) − log(rank_p2)         |
| 37 | `rank_pts_diff` | log(rank_pts_p1) − log(rank_pts_p2) |
| 38 | `age_diff`      | age_p1 − age_p2                     |
| 39 | `ht_diff`       | ht_p1 − ht_p2                       |
| 40 | `form_diff`     | form_p1 − form_p2                   |

---

## 6. Logika działania — krok po kroku

### ETAP 1. Wczytanie i przygotowanie danych

```
2024.csv → DataFrame → parsowanie tourney_date → sortowanie chronologiczne
         → selekcja kolumn (cols_base) → dropna() → transformacje (log, is_lefty, round)
```

1. Wczytanie pliku `sample_data/2024.csv` (3076 meczów surowych).
2. Parsowanie daty turnieju (`tourney_date`) i sortowanie chronologiczne wg `(tourney_date, match_num)`.
3. Selekcja kolumn bazowych (`cols_base` = 34 kolumny: kontekst + atrybuty graczy + statystyki serwisowe).
4. `dropna()` — usunięcie meczów z brakującymi danymi (~126 meczów: walkowery, retirements bez statystyk). Wynik: **2950 meczów**.
5. Transformacje:
   - `log(rank)` i `log(rank_points)` — kompresja prawoskośnego rozkładu rankingów
   - Binaryzacja ręczności: `'L'` → `1`, `'R'` → `0`
   - Ordynalne kodowanie rundy: R128=1, R64=2, ..., F=7

### ETAP 2. Dane historyczne (redukcja cold-start)

```
2018.csv + 2019.csv + ... + 2023.csv → concat → 14 945 meczów historycznych
```

- Problem: Na początku sezonu 2024 model nie zna żadnego gracza — forma = 0.5, H2H = 0, statystyki serwisowe = domyślne ATP.
- Rozwiązanie: Wczytanie **6 sezonów historii** (2018–2023) jako bazy do obliczania cech dynamicznych. Każdy gracz ma setki meczów „za sobą" zanim model zacznie predykcje na 2024.

### ETAP 3. Kodowanie zmiennych kategorycznych

- **LabelEncoder** dopasowywany na połączonym zbiorze (2018–2024), aby uwzględnić wszystkie możliwe kategorie.
- Kodowane zmienne: `surface` (3 kategorie) i `tourney_level` (8 kategorii).
- LabelEncoder generuje fałszywą ordynalność (Clay=0 < Grass=1 < Hard=2), ale **nie szkodzi** Random Forestowi — drzewa decyzyjne używają podziałów progowych, nie arytmetyki na wartościach.

### ETAP 4. Podział chronologiczny (60/20/20)

```
[--- 60% trening ---][--- 20% walidacja ---][--- 20% test ---]
      1770 meczów          590 meczów          590 meczów
   styczeń–lipiec 2024    lipiec–październik     październik–grudzień
```

- **Dlaczego chronologiczny, a nie losowy?** Losowy podział (np. `train_test_split(shuffle=True)`) prowadziłby do **data leakage** — model mógłby trenować na meczach z października i testować się na meczach z marca, „widząc przyszłość". Podział chronologiczny gwarantuje, że model zawsze testowany jest na danych nowszych niż treningowe.

### ETAP 5. Cechy dynamiczne (Expanding Window)

```
Dla meczu i:
  historia = dane_historyczne (2018–2023) + mecze_z_bieżącego_zbioru[0..i-1]
  → oblicz formę, H2H, statystyki serwisowe na podstawie WYŁĄCZNIE przeszłości
```

Kluczowa zasada: **żaden mecz nie widzi swojej przyszłości**. Dla i-tego meczu w zbiorze:

- **Historia** = wszystkie mecze z sezonów 2018–2023 + mecze 0..i-1 z bieżącego zbioru
- Obliczane cechy:

  - **Forma** (`calculate_form`): win rate z ostatnich 10 meczów danego gracza
  - **Forma nawierzchniowa** (`calculate_surface_form`): win rate z ostatnich 10 meczów na tej nawierzchni; jeśli gracz ma <3 mecze na nawierzchni → fallback na formę ogólną
  - **Head-to-Head** (`get_h2h`): liczba wygranych p1 − wygranych p2 w bezpośrednich spotkaniach z całej historii
  - **8 statystyk serwisowych** (`calculate_serve_stats`): rolling average z 10 ostatnich meczów (wskaźniki procentowe)
- Historia narasta kaskadowo:

  - **Trening**: historia = 2018–2023
  - **Walidacja**: historia = 2018–2023 + zbiór treningowy 2024
  - **Test**: historia = 2018–2023 + zbiór treningowy 2024 + zbiór walidacyjny 2024

### ETAP 6. Symetryzacja danych

```
Mecz: Djokovic pokonuje Nadala
  → Wiersz 1: P1=Djokovic, P2=Nadal,  y=1 (P1 wygrywa)
  → Wiersz 2: P1=Nadal,    P2=Djokovic, y=0 (P1 przegrywa)
```

- **Problem**: Dane CSV zawsze umieszczają zwycięzcę w kolumnach `winner_*`. Gdyby model trenował bezpośrednio, nauczyłby się trywialnej reguły „P1 zawsze wygrywa" → accuracy 100% na treningu, 50% na teście.
- **Rozwiązanie**: Każdy mecz generuje **2 symetryczne przykłady treningowe**:
  - Perspektywa 1: P1 = zwycięzca → y = 1
  - Perspektywa 2: P1 = przegrany → y = 0
- W perspektywie 2 wszystkie cechy zależne od gracza są **zamieniane** (rank, forma, serwis), a cechy różnicowe i H2H — **negowane**.
- Efekt: rozkład etykiet jest idealnie zbalansowany (50:50), a model uczy się „kto wygra", nie „czy P1 wygra".

### ETAP 7. Definicja wektora cech

Lista 40 cech przekazywanych do algoritmu (patrz [Sekcja 5](#5-pełna-lista-cech-modelu-40)).

### ETAP 8. Optymalizacja hiperparametrów

```
RandomizedSearchCV(
  estimator = RandomForestClassifier,
  n_iter    = 50,       ← 50 losowych kombinacji hiperparametrów
  cv        = TimeSeriesSplit(5),  ← 5 foldów chronologicznych
  scoring   = 'accuracy'
)
```

- **Przestrzeń przeszukiwania**:

  - `n_estimators`: [100, 200, 300, 500] — liczba drzew w lesie
  - `max_depth`: [10, 15, 20, 30, None] — maksymalna głębokość drzewa
  - `min_samples_split`: [2, 5, 10, 20] — min. próbek do podziału węzła
  - `min_samples_leaf`: [1, 2, 4, 8] — min. próbek w liściu
  - `max_features`: ['sqrt', 'log2'] — ile cech losować przy każdym podziale
  - `max_samples`: [0.7, 0.8, 0.9, 1.0] — odsetek danych w każdym drzewie (regularyzacja)
- **TimeSeriesSplit** dzieli dane na 5 foldów z zachowaniem chronologii:

  ```
  Fold 1: train=[0..N/5]      val=[N/5..2N/5]
  Fold 2: train=[0..2N/5]     val=[2N/5..3N/5]
  ...
  Fold 5: train=[0..4N/5]     val=[4N/5..N]
  ```

  W żadnym foldzie dane walidacyjne nie „cofają się" do przeszłości.
- Dane treningowe dla CV **nie są mieszane** (`shuffle=False`), ponieważ TimeSeriesSplit wymaga zachowania porządku czasowego.

### ETAP 9. Trening finalnego modelu

- Po znalezieniu najlepszych hiperparametrów, finalny model trenowany jest na **pełnym zbiorze treningowym** (z `shuffle=True` — bo nie robimy już walidacji krzyżowej, a Random Forest korzysta z wewnętrznego baggingu).

### ETAP 10. Ewaluacja

- **Na danych symetrycznych**: accuracy, classification report, confusion matrix.
- **Na poziomie meczów**: z każdej pary symetrycznej bierzemy perspektywę zwycięzcy (y=1, gdzie P1 = rzeczywisty zwycięzca). Sprawdzamy, czy model przypisał P1 prawdopodobieństwo > 0.5.

---

## 7. Techniki i „sztuczki" poprawiające model

### 7.1. Symetryzacja — eliminacja positional bias

Najważniejsza innowacja. Bez niej model widzi 100% etykiet y=1 (bo P1 = winner z definicji CSV). Symetryzacja zmienia to w problem binarnej klasyfikacji z rozkładem 50:50.

### 7.2. Transformacja logarytmiczna rankingów i punktów

Ranking ATP ma rozkład prawoskośny (nieliniowy). `log(rank)` lepiej oddaje relację siła → ranking. Różnica GS #1 vs #10 (log: 0 vs 2.3) jest proporcjonalnie większa niż #100 vs #110 (log: 4.6 vs 4.7), co odpowiada rzeczywistości tenisowej.

### 7.3. Dane historyczne 2018–2023 (redukcja cold-start)

Bez historii na początku sezonu 2024 wszystkie cechy dynamiczne startują od wartości domyślnych. Z 6-letnią historią (14 945 meczów) nawet mało znani gracze mają kilkadziesiąt meczów w „pamięci" modelu.

### 7.4. Expanding window (brak data leakage)

Cechy dynamiczne obliczane są wyłącznie z meczów PRZED bieżącym meczem. Model nigdy nie „widzi" przyszłości — ani wyników, ani statystyk z późniejszych meczów.

### 7.5. Podział chronologiczny (brak temporal leakage)

Dane dzielone chronologicznie (nie losowo). Model testowany na meczach z końcówki sezonu 2024, które nigdy nie były widziane podczas treningu ani walidacji krzyżowej.

### 7.6. TimeSeriesSplit w CV

Zamiast standardowej walidacji krzyżowej (k-fold), która losowo miesza dane, `TimeSeriesSplit` dzieli je z zachowaniem chronologii. To eliminuje ryzyko „trenowania na przyszłości" nawet wewnątrz procesu doboru hiperparametrów.

### 7.7. Cechy różnicowe (rank_diff, ht_diff itd.)

Oprócz cech absolutnych (rank_p1, rank_p2), model ma cechy różnicowe (rank_p1 − rank_p2). Pozwala to drzewom na prosty podział „jeśli rank_diff < −2 → P1 wygrywa" zamiast uczenia się złożonych interakcji dwóch rang.

### 7.8. Fallback w formie nawierzchniowej

Jeśli gracz rozegrał <3 mecze na danej nawierzchni, model „wraca" do formy ogólnej zamiast bazować na niereprezentatywnej próbce.

### 7.9. Wartości domyślne (prior) dla cold-start

Gracze bez historii otrzymują domyślne statystyki ATP (średnie tourowe), co stabilizuje predykcje i zapobiega ekstremalnym wartościom.

### 7.10. Ordynalne kodowanie rund

Rundy kodowane ordynalnie (R128=1 → F=7) zamiast arbitralnie, co zachowuje naturalną hierarchię: im dalej w turnieju, tym wyższa wartość. Random Forest potrafi z tego skorzystać.

---

## 8. Wyniki modelu

| Metryka                                  | Wartość            |
| ---------------------------------------- | -------------------- |
| CV Accuracy (TimeSeriesSplit, 5 foldów) | ~0.648               |
| Validation Accuracy (dane symetryczne)   | ~0.633               |
| Test Accuracy (dane symetryczne)         | ~0.614               |
| **Match Prediction Accuracy**      | **~61.0%**     |
| Baseline (losowe zgadywanie)             | 50.0%                |
| **Przewaga nad baseline**          | **+11.0 p.p.** |

> Wyniki mogą nieznacznie różnić się między uruchomieniami ze względu na losowość `RandomizedSearchCV`. Ziarno `RANDOM_STATE = 42` zapewnia powtarzalność przy identycznym zbiorze danych.

### Kontekst wyniku

- W literaturze naukowej modele predykcji meczów ATP osiągają typowo **60–67% accuracy** w zależności od użytych danych (np. kursy bukmacherskie dają ~68%).
- 61% bez danych bukmacherskich, z samymi statystykami meczowymi, to wynik **solidny i w normie akademickiej**.

---

## 9. Ważność cech (Feature Importance)

Poniższa tabela przedstawia ważność cech mierzoną jako **Mean Decrease Impurity** (średni spadek zanieczyszczenia Giniego) uśredniony po wszystkich drzewach lasu.

| Pozycja | Cecha                    | Importance | Kategoria   |
| :-----: | ------------------------ | :--------: | ----------- |
|    1    | `rank_pts_diff`        |   0.107   | Różnicowa |
|    2    | `rank_diff`            |   0.085   | Różnicowa |
|    3    | `p1_rank_pts_log`      |   0.049   | Statyczna   |
|    4    | `p2_rank_log`          |   0.047   | Statyczna   |
|    5    | `p1_rank_log`          |   0.044   | Statyczna   |
|    6    | `p2_rank_pts_log`      |   0.043   | Statyczna   |
|    7    | `p1_first_won_pct`     |   0.035   | Serwisowa   |
|    8    | `p2_first_won_pct`     |   0.033   | Serwisowa   |
|    9    | `p1_second_won_pct`    |   0.030   | Serwisowa   |
|   10   | `p1_bp_faced_per_game` |   0.030   | Serwisowa   |
|   ...   | ...                      |    ...    | ...         |
|   37   | `surface`              |   0.005   | Kontekstowa |
|   38   | `p1_is_lefty`          |   0.002   | Statyczna   |
|   39   | `best_of`              |   0.002   | Kontekstowa |
|   40   | `p2_is_lefty`          |   0.001   | Statyczna   |

### Interpretacja

- **Ranking i punkty rankingowe dominują** — 6 z 10 najważniejszych cech to cechy rankingowe.
- **Statystyki serwisowe mają umiarkowany wpływ** — `first_won_pct` i `second_won_pct` (efektywność 1. i 2. serwisu) są najważniejsze z grupy serwisowej.
- **H2H, forma i nawierzchnia mają niski wpływ** — co nie oznacza, że są bezwartościowe. Random Forest potrafi wykorzystać je w interakcji z innymi cechami (np. forma + nawierzchnia → forma na danej nawierzchni jest obejmowana przez `surface_form`).
- **Ręczność i best_of mają najniższy wpływ** — oczekiwane, ponieważ left-handedness jest rzadka (~14% graczy), a best_of jest silnie skorelowane z `tourney_level` (tylko Grand Slamy grają best-of-5).

---

## 10. Ograniczenia i uwagi

### 10.1. Brak danych bukmacherskich

Kursy bukmacherskie (np. Pinnacle, Bet365) są najsilniejszym indywidualnym predyktorem wyników meczów tenisowych (~68% accuracy). Model celowo ich nie wykorzystuje, polegając wyłącznie na statystykach meczowych.

### 10.2. Brak systemu Elo

Model nie implementuje systemu ocen Elo (ani żadnej jego wariacji — Glicko, TrueSkill). System Elo aktualizuje rating graczy po każdym meczu i historycznie dobrze predykuje tenis. Dodanie Elo mogłoby poprawić wynik o 1–3 p.p.

### 10.3. Goldne standy losowości

`RANDOM_STATE = 42` zapewnia powtarzalność, ale wynik jest zależny od konkretnego ziarna. Różne ziarna mogą dać wyniki w zakresie ±1.5 p.p. (normalny szum statystyczny z 590 meczami testowymi).

### 10.4. Sezonowość i kontuzje

Model nie ma danych o:

- Kontuzjach i przerwach w grze (gracz wracający po 6-miesięcznej kontuzji ma stare, nieaktualne statystyki)
- Warunkach pogodowych (wiatr, temperatura, wilgotność)
- Zmęczeniu turniejowym (ile meczów gracz rozegrał w ostatnich dniach)
- Strefie czasowej i jet-lagu

### 10.5. Złożoność obliczeniowa

Implementacja cech dynamicznych z expanding window ma złożoność O(n²) — dla każdego meczu skanuje całą historię. Przy 2950 meczach × 14 945 historycznych jest to akceptowalne (minuty), ale niewydajne przy skalowaniu. Alternatywą byłby model EWMA z przyrostową aktualizacją stanu (O(n)).

### 10.6. LabelEncoder vs OneHotEncoder

LabelEncoder wprowadza fałszywą ordynalność (Clay=0 < Grass=1 < Hard=2). Dla drzew decyzyjnych (Random Forest) nie stanowi to problemu, ale zamiana na OneHotEncoder mogłaby minimalnie poprawić wynik dla modeli liniowych.
