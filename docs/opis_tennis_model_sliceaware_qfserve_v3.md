# Opis pliku `tennis_model_sliceaware_qfserve_v3.py` — wariant QF + warunkowy serwis

> Wszystkie pojęcia techniczne są szczegółowo wyjaśnione w `SLOWNICZEK_POJEC.md`. Tu są skrócone wytłumaczenia inline.

## Co ten plik robi w jednym zdaniu
Najbogatszy wariant slice-aware (~50 nowych cech). Łączy 3 kierunki: cechy turniejowe (seed gracza, siła już pokonanych rywali w drabince), warunkowy serwis (per nawierzchnia, vs lewo/prawo, pod presją), oraz wszystkie cechy z prostego `sliceaware`. Drugi najlepszy wynik (+2.20 p.p.).

## Główne założenia

1. **Kontekst turniejowy** — informacja, której baseline w ogóle NIE MA:
   - **Seed gracza** (rozstawienie w drabince) — pre-match informacja o tym, że organizator uznał go za faworyta.
   - **Tournament path opponent strength** — średnia siła rywali, których gracz już pokonał w bieżącym turnieju. Mówi: „gracz dotarł do QF łatwą czy trudną drogą".
   - **Draw size** — z ilu graczy startował turniej (128 dla Grand Slamu, 32 dla ATP 250).

2. **Warunkowy serwis (Serve v2)** — serwis nie jest stałą cechą gracza. Gracz może mieć potężny serwis na hard court i przeciętny na clay. Dlatego liczymy 4 warianty profilu serwisowego: surface, vs top opponents, vs hand, pressure (Bo5 / late rounds).

3. **Walidacja `tourney_id`** — assercja że ma format `YYYY-...` (np. `2024-580`). Bez tego `tourney_path_*` mógłby leakować dane międzyletnie (różne turnieje z tym samym numerycznym ID).

4. **Seed fallback przez ranking** — większość graczy w danych Jeff Sackmana ma `seed=NaN` (rozstawiani są tylko top 8-32). Gdy seed jest pusty, używamy proxy z rankingu: niski ranking → wysoki seed_context_score.

5. **Wszystkie cechy ze sliceaware** plus nowe — to v3, najbogatszy wariant. Bierze dobre rzeczy z poprzednich iteracji + dodaje seed/path/conditional-serve.

## Dlaczego takie wybory, a nie inne?

- **Czemu seed jako pre-match info?** Bo on jest USTALANY PRZED turniejem na podstawie aktualnego rankingu. Top seedy są chronieni od siebie w drabince — to zewnętrzna ocena „kto jest faworytem". Model dostaje to za darmo.
- **Czemu opp_strength przez log1p?** Punkty rankingowe ATP są mocno prawoskośne (Djokovic miał ~10000 vs ostatni gracz ~50). Log1p ściska skalę i daje sensowne średnie.
- **Czemu top_opp_threshold zależy od poziomu turnieju?** Na Grand Slamie i Mastersie „top 20" to znaczący próg. Na ATP 250 grają głównie zawodnicy spoza top 30 — tam threshold 40 ma więcej sensu (bo inaczej slice byłby pusty).
- **Czemu pressure_serve = Bo5 OR late rounds?** Bo presja może wynikać z dwóch rzeczy: długi format (Bo5) ALBO ważny etap turnieju (QF/SF/F na Bo3). Łączymy oba.
- **Czemu walidacja regex tourney_id?** Jeff Sackmann używa formatu `YYYY-XXX`. Gdyby ktoś podmienił dane (np. samo `XXX`), `tourney_path_*` matchowałby Wimbledon 2018 z Wimbledon 2024 — leakage. Assert chroni przed cichym błędem.

## Słowniczek pojęć z tego pliku

| Pojęcie | Co znaczy |
|---|---|
| **seed (rozstawienie)** | Numer rozstawienia gracza w drabince (1=top seed, im wyższy numer tym słabszy). Tylko top 8-32 ma seedy. |
| **seed_context_score** | Znormalizowany seed do [0,1]. 1.0 = top seed, ~0 = najsłabszy. Fallback z rankingu gdy NaN. |
| **draw size** | Liczba graczy w turnieju (128/64/32). Grand Slam=128, Masters=64-96, ATP 250=32. |
| **tourney_path_opp_strength** | Średnia siła rywali pokonanych przez gracza w BIEŻĄCYM turnieju. Mówi „łatwa droga vs trudna droga do QF". |
| **conditional serve / serve v2** | Profil serwisowy w konkretnym kontekście (powierzchnia / vs hand / pod presją), nie ogólna średnia. |
| **log1p** | `ln(1+x)` — kompresja skali, działa dla x=0. Używamy dla skoszonych rozkładów (rank_points). |
| **pressure** | Bo5 OR (round in {QF, SF, BR, F}). Dwa źródła presji łączone razem. |
| **tourney_id leakage** | Risk gdy tourney_id nie ma roku — różne turnieje z tym samym numerem matchują przez lata. Walidacja regex chroni. |
| **interakcja (qf_level_pressure)** | Cecha = inna cecha × inna cecha. RF widzi gotową kombinację, nie musi sam ją odkrywać. |

## Ważne metody (1:1 z kodu)

| Metoda | Co robi |
|---|---|
| `PlayerHistoryIndex` (klasa) | Identyczna jak w bestof5_v1 — bisect na sorted indeksach. |
| `validate_tourney_id_format(frame, source)` | Sprawdza assertion `tourney_id` matchuje `^\d{4}-` — wyrzuca ValueError jeśli format jest nieprawidłowy. |
| `set_history_context(...)`, `get_player_history(...)` | Module-level state pattern jak w bestof5_v1. |
| `filter_player_history(...)` | Bierze historię gracza i dorzuca filtry: best_of, rounds, surface, opponent_hand, opponent_rank_max. 5 osi filtrowania. |
| `calculate_context_form(...)` | Identyczna idea jak sliceaware — forma w kontekście z fallbackiem. |
| `calculate_context_experience(...)` | Doświadczenie w kontekście. |
| `calculate_context_balance(...)` | Bilans wins-losses vs konkretna ręczność. |
| `extract_player_match_serve_metrics(...)`, `compose_serve_score(...)`, `build_fallback_serve_profile(...)` | Tak samo jak w bestof5_v1 — wyciąganie 8 stats serwisu + agregacja w skalar. |
| `calculate_context_serve_profile(...)` | Profil serwisowy w warunkach — z 5 filtrami (best_of, rounds, surface, opponent_hand, opponent_rank_max). |
| `estimate_seed_slots(draw_size)` | Mapuje rozmiar drabinki na liczbę slotów seedingu: draw=128 → 32 seedy, draw=64 → 16 seedów, draw=32 → 8 seedów. |
| `compute_seed_context_score(seed_value, rank, draw_size)` | Liczy „seed score" [0, 1]. Top seed → 1.0, najniższy seed → ~0.0, NaN → soft proxy z rankingu. |
| `strong_opponent_threshold(tourney_level)` | Próg „mocnego rywala" zależnie od poziomu turnieju: G/M=20, 500/A/F=30, inne=40. |
| `tournament_level_strength(level)` | Mapuje literę poziomu → liczba (G=1.00, 250=0.68 itd.). |
| `opponent_rank_points(match, player_name)` | Wyciąga punkty rankingowe rywala z danego meczu (zależnie od tego czy gracz był winnerem czy loserem). |
| `calculate_tournament_path_stats(player_name, current_row, past_matches)` | KLUCZOWA cecha — filtruje mecze tego samego `tourney_id` (czyli bieżącego turnieju), bierze tylko mecze gracza, liczy średnią siłę rywali (log1p rank_points) i liczbę meczów. Mówi modelowi: „w QF Wimbledonu gracz A pokonał #50→#30→#20 — to lepsza droga niż gracz B który pokonał #200→#150→#100". |
| `pressure_serve_profile(...)` | Inteligentny wybór profilu serwisowego: jeśli Bo5 → Bo5 profil, jeśli late round → late round profil, inaczej fallback. |
| `add_targeted_slice_features(...)` | Główna pętla — ~25 cech kontekstowych na mecz. |
| `attach_targeted_features(...)` | Mapping + interakcje: `qf_level_pressure = is_qf × tourney_level_strength`, `best_of5_level_pressure = is_best_of5 × tourney_level_strength`. |
| `run_sliceaware_qfserve_v3()` | Funkcja main. |

## Ważne zmienne

| Zmienna | Co oznacza |
|---|---|
| `TOURNEY_ID_PATTERN = re.compile(r"^\d{4}-")` | Regex walidacyjny — wymaga prefiksu roku. |
| `PRESSURE_ROUNDS = LATE_ROUNDS = {"QF", "SF", "BR", "F"}` | Zbiór rund presyjnych. Identyczny set, dwie nazwy dla czytelności. |
| `EXTRA_CONTEXT_COLUMNS` | 5 doczytywanych kolumn: tourney_id, tourney_name, draw_size, winner_seed, loser_seed. |
| `TOURNEY_LEVEL_STRENGTH` | Słownik mapowania liter poziomu turnieju na liczby. |
| `TARGETED_FEATURES` | Lista ~50 nowych cech. |
| `SYMMETRIC_FEATURE_SPECS` | 18 par (feature, diff) — definiuje co symetryzować. |
| `_HISTORY_INDEX`, `_HISTORY_CUTOFF` | Module-level state pattern. |
| `winner_path_stats`, `loser_path_stats` | Dict z `opp_strength` i `match_count` dla bieżącego turnieju. |
| `winner_surface_serve`, `winner_top_opp_serve`, `winner_vs_opp_hand_serve`, `winner_pressure_serve` | 4 warianty profilu serwisowego dla winner. To samo dla loser. |
| `top_opp_threshold` | Próg „mocnego rywala" dla bieżącego meczu — zależy od poziomu turnieju. |
| `match_accuracy` | **63.22%** — drugi najlepszy wynik (za bestof5_v1). |

## Wyniki

- Match accuracy: **63.22%** vs baseline **61.02%**
- **Delta: +2.20 p.p.** — drugi najlepszy wariant
- SPEKTAKULARNE zyski na slice'ach które inni mocno przegrywali:
  - `round=R128 × L-vs-R` (R128 GS z lewo): baseline 33.3% → **77.8%** (**+44.4 p.p.**)
  - `round=R128 × rank_gap=51-100`: +40.0 p.p.
  - `round=R128 × rank_gap=11-25`: +33.3 p.p.
  - `tourney_level=250 × QF`: +7.4 p.p. (faktyczna poprawa QF)
- Cechy `seed_context_diff`, `tourney_path_opp_strength_diff` wchodzą w top 30 importance — model je wykorzystuje
- Spadki podobne jak bestof5_v1 (finały, małe support)

**Wniosek**: dodanie informacji ZEWNĘTRZNYCH dla modelu (seed, drabinka, kontekst turniejowy) daje większą wartość niż tylko historyczne statystyki gracza. To miejsce gdzie warto by jeszcze inwestować.

## Co odpowiedzieć gdy promotor zapyta…

**Q: „Co to jest 'seed' i czemu jest ważny?"**
A: Seed to numer rozstawienia gracza w drabince turnieju — organizator ustala go PRZED turniejem na podstawie aktualnego rankingu. Top 8-32 seedy są CHRONIENI od siebie (Nadal i Djokovic nie zagrają przed półfinałem). Seed niesie informację „organizator uważa go za faworyta", której nie ma w samym rankingu (np. Murray po kontuzji może mieć ranking 80 ale seed 16 z dziką kartą).

**Q: „Co to jest 'tournament path opponent strength'?"**
A: Średnia siła rywali, których gracz już pokonał w BIEŻĄCYM turnieju, dochodząc do bieżącej rundy. Dwóch graczy doszło do QF: A pokonał #50→#30→#20, B pokonał #200→#150→#100. A miał trudniejszą drogę → A jest lepszy niż mówi jego ranking. Ta cecha mówi modelowi „popatrz na drogę, nie tylko na ranking".

**Q: „Czemu walidujecie format tourney_id przez regex?"**
A: Bo Jeff Sackmann używa formatu `YYYY-XXX` (np. `2024-580`). Gdyby przy doczytywaniu z innego źródła pojawiło się samo `580`, nasza funkcja `calculate_tournament_path_stats` matchowałaby ten sam turniej między latami — Wimbledon 2018 z Wimbledon 2024. To data leakage. Regex `^\d{4}-` chroni przed takim cichym błędem.

**Q: „Co to jest 'conditional serve' (serwis warunkowy)?"**
A: Profil serwisowy zależnie od warunków. Gracz może serwować świetnie na hard court (ace rate 15%), ale słabo na clay (8%). Albo świetnie vs leworęcznych, ale przeciętnie vs praworęcznych. Liczymy 4 warianty: per surface, vs top opponents, vs opponent_hand, pod presją (Bo5/late round). To jest „serve v2" w przeciwieństwie do baseline'owego serwisu (jedna ogólna średnia).

**Q: „Co to jest log1p i czemu go używacie?"**
A: `log1p(x) = ln(1+x)`. Działa dla x=0 (zwraca 0, nie -∞). Punkty rankingowe ATP są mocno prawoskośne — Djokovic ma ~10000, ostatni gracz ~50. Surowa średnia by była zdominowana przez topowych. Log1p ściska skalę → sensowna średnia.

**Q: „Czemu pressure = Bo5 OR (round QF/SF/F)?"**
A: Bo presja może wynikać z dwóch rzeczy. Bo5 (długi format) wymaga wytrzymałości i koncentracji 4-5 godzin. Late round (QF/SF/F) wiąże się z tytułem turnieju, presją mediów, mniej meczów do końca turnieju. Niezależne źródła presji — łączymy OR. Model ma cechę `is_best_of5` i `late_round_flag` — sam decyduje czy łączyć.

**Q: „Czemu seed_context_score, a nie surowy seed?"**
A: Bo seed=1 vs seed=32 znaczy co innego dla drawa 128 (32 seedy → seed=1 czołówka) i drawa 32 (8 seedów → seed=1 absolutna czołówka). Normalizujemy do [0,1] przez relację „pozycja seeda / liczba slotów seedingu". Plus fallback z rankingu gdy seed=NaN (większość graczy nie jest rozstawiona).

**Q: „Czemu top_opp_threshold zależy od poziomu turnieju?"**
A: Bo „mocny rywal" znaczy co innego. Na Grand Slamie i Mastersie top 20 to znaczący próg (większość pola spoza top 100 i tak nie dojdzie do późnych rund). Na ATP 250 cała stawka to często gracze spoza top 30 — gdyby threshold był 20, slice byłby pusty. Dynamicznie dostosowujemy.

**Q: „Czemu interakcje (qf_level_pressure, best_of5_level_pressure) explicit?"**
A: Bo RF mógłby je nauczyć sam, ale każde drzewo ma ograniczoną głębokość. Explicit interakcja = jedna cecha dla RF, gotowy sygnał. Sprawdziliśmy że wchodzą w top 30 feature importance — model je faktycznie wykorzystuje.

**Q: „Czemu drugi najlepszy, a nie pierwszy?"**
A: BestOf5_v1 ma 63.39%, my mamy 63.22%. Różnica 0.17 p.p. — w obrębie szumu eksperymentalnego. Ale qfserve v3 ma więcej cech (50 vs 37) — większa złożoność za marginalną poprawę. Bo5 wygrywa bo focused approach z dobrze zdefiniowanym kompozytem (endurance_score). Lekcja: prostota wygrywa z bogactwem.
