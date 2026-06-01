# Opis pliku `main_48_cech_sliceaware.py` — szeroki wariant slice-aware

> Wszystkie pojęcia techniczne są szczegółowo wyjaśnione w `SLOWNICZEK_POJEC.md`. Tu są skrócone wytłumaczenia inline.

## Co ten plik robi w jednym zdaniu
Bierze baseline z `main_48_cech.py`, dodaje 20 cech kontekstowych mających pomóc w trzech najsłabszych slice'ach naraz (Bo5, QF, leworęczny-vs-praworęczny) i trenuje nowy Random Forest.

## Główne założenia

1. **Shotgun approach** — strzelamy po szerokiej grupie problemów naraz. Wykryliśmy 3 słabe podgrupy w `modelslice` — to zróbmy cechy dla wszystkich trzech jednocześnie. „Shotgun" = strzał ze śrutem (wiele celów), w przeciwieństwie do „sniper" (jeden cel).

2. **Reuse baseline** — nie zmieniamy hiperparametrów modelu, nie zmieniamy splitu, nie zmieniamy istniejących cech. Po prostu dorzucamy NOWE cechy obok 40 baseline'owych. Tak żeby porównanie było uczciwe — jakakolwiek delta accuracy wynika z cech, a nie z innego tuningu (ablation study).

3. **Expanding window** — wszystkie nowe cechy liczone TYLKO z meczów ROZEGRANYCH przed bieżącym. Bez tej zasady byłoby leakage danych z przyszłości. To samo co w baseline'ie.

4. **Fallback do ogólnej formy** — jak gracz nie ma minimum 2-3 meczów w danym kontekście (np. nigdy nie grał Bo5), to zamiast neutralnego 0.5 zwracamy jego ogólną formę. Lepszy szacunek niż „nic nie wiem".

5. **Index gracz → wiersze, bisect** — optymalizacja prędkości. Bez indexu, każda cecha dla każdego meczu skanowałaby 18 000 wierszy historii. Index sprowadza to do O(log K).

## Dlaczego takie wybory, a nie inne?

- **Czemu 20 cech, a nie więcej / mniej?** Każdy z 3 słabych slice'ów dostaje ~3-4 dedykowane cechy (form, experience, surface form, balance). Mniej = ryzyko że RF nie zauważy. Więcej = ryzyko rozcieńczenia sygnału.
- **Czemu fallback do ogólnej formy a nie do 0.5?** 0.5 to „nic nie wiem", co dla nowych graczy wciągałoby ich do średniej. Ogólna forma to „wiem coś o graczu, ale nie w tym konkretnym kontekście" — lepszy proxy.
- **Czemu PlayerHistoryIndex + bisect?** Bez tego każda cecha dla każdego meczu skanowała całą historię (~18 000 wierszy). 16 cech × 2700 meczów × 18 000 = setki milionów porównań stringów = pipeline ciągnie się minuty. Index sprowadza lookup do O(log K) — przyspiesza ~5-10x.
- **Czemu nie zmieniamy hiperparametrów RF?** Bo chcemy porównać UCZCIWIE: baseline vs sliceaware. Jakakolwiek różnica musi pochodzić od NOWYCH CECH, a nie od inaczej dobranych hiperparametrów. To nazywamy „ablation study" — izolujemy efekt jednej zmiany.

## Słowniczek pojęć z tego pliku

| Pojęcie | Co znaczy |
|---|---|
| **shotgun approach** | Dodanie wielu cech jednocześnie, atakujących różne problemy. Przeciwieństwo: focused (jeden problem na raz). |
| **slice-aware** | Model „świadomy" istnienia konkretnych podgrup — dostaje cechy specyficzne dla nich. |
| **expanding window** | Cecha liczy się tylko z meczów PRZED bieżącym (nie widzi przyszłości). |
| **fallback** | Wartość zastępcza gdy brak danych. U nas: ogólna forma zamiast 0.5. |
| **late_rounds** | QF, SF, BR, F — etapy gdzie presja jest największa. |
| **opponent_hand** | Ręczność rywala (L=lewa, R=prawa). Cechy „forma vs L" / „forma vs R" pomagają w slice L-vs-R. |
| **bisect / bisect_left** | Funkcja Pythona do binarnego wyszukiwania w posortowanej liście. O(log n). |
| **ablation study** | Eksperyment, gdzie zmienia się tylko jedną rzecz, żeby zobaczyć JEJ konkretny efekt. |

## Ważne metody (1:1 z kodu)

| Metoda | Co robi |
|---|---|
| `execute_base_pipeline_quietly()` | Uruchamia `main_48_cech.py` z wyciszonym outputem (przekierowanie stdout do bufora). Zwraca namespace baseline'u. |
| `build_player_index(full_sequence)` | Buduje mapę `gracz → sorted lista indeksów wierszy gdzie wystąpił`. Liniowy skan pełnej historii RAZ. |
| `get_player_history_via_index(player, full_sequence, player_index, cutoff)` | Pobiera mecze gracza ROZEGRANE przed `cutoff` (czyli wcześniej chronologicznie). Używa `bisect_left` na posortowanej liście indeksów. O(log K) zamiast O(N). |
| `get_player_history(player_name, history)` | Stara, wolna wersja — filtruje całą historię pandasem. Trzymana dla legacy callerów. |
| `_apply_context_filters(player_history, ...)` | Bierze już-przefiltrowaną historię gracza i dorzuca filtry: best_of, rounds, surface, opponent_hand. |
| `calculate_context_form(...)` | Liczy formę gracza w konkretnym kontekście (np. tylko Bo5, tylko QF). Window=12 ostatnich meczów w tym kontekście. Fallback do ogólnej formy gdy < min_matches. |
| `calculate_context_experience(...)` | Ile gracz rozegrał meczów w danym kontekście, znormalizowane do [0,1] przez `scale`. Mówi czy gracz jest „weteranem Bo5" czy „początkującym". |
| `calculate_context_balance(...)` | Bilans wins-losses przeciwko konkretnej ręczności, znormalizowany. Wartość dodatnia = gracz lubi rywali tej ręczności. |
| `add_targeted_slice_features(df_subset, historical_data, base_cols)` | Główna pętla — dla każdego meczu liczy 20 cech kontekstowych dla winner i loser. Buduje index na początku. |
| `attach_targeted_features(symmetrized_data, raw_data)` | Po symetryzacji mapuje `w_*`/`l_*` (perspektywa zwycięzcy/przegranego) na `p1_*`/`p2_*` (perspektywa nowa) używając `np.where(y==1, ...)`. Dolicza też binarne flagi: `is_best_of5`, `is_qf`, `is_lefty_matchup`. |
| `print_metric_delta(name, baseline_value, new_value)` | Drukuje porównanie typu „Validation: baseline=0.6331 | slice-aware=0.6322 | delta=-0.0009". |
| `run_sliceaware_model()` | Funkcja main — uruchamia całość i drukuje wyniki. |

## Ważne zmienne

| Zmienna | Co oznacza |
|---|---|
| `BASE_SCRIPT` | Ścieżka do `main_48_cech.py`. |
| `LATE_ROUNDS = {"QF", "SF", "BR", "F"}` | Zbiór „późnych rund turnieju" — gdzie zwykle decydują się tytuły i presja jest największa. |
| `TARGETED_FEATURES` | Lista 32 nazw nowych cech (z is_best_of5, is_qf, is_lefty_matchup włącznie). Używana do feature importance i jako rozszerzenie `features` baseline'u. |
| `player_index` | Mapa `nazwa → sorted lista indeksów` dla całej historii. Liczona raz, używana na każde wywołanie. |
| `df_train_raw`, `df_val_raw`, `df_test_raw` | Te same splity co baseline, ale z dorobionymi kolumnami `w_best_of5_form`, `l_best_of5_form` itd. |
| `train_data`, `val_data`, `test_data` | Po symetryzacji + attach_targeted_features — gotowe do treningu, z `p1_*`/`p2_*` formą cech. |
| `features` | Lista nazw cech: baseline (40) + TARGETED_FEATURES (32) = 72 cechy. |
| `best_rf` | Nowy RF wytrenowany na 72 cechach, ALE z hiperparametrami baseline'u. |
| `match_accuracy` | Match accuracy tego wariantu (u mnie: **60.85%**, baseline: 61.02%). |
| `feature_importance` | Sortowana tabela ważności cech. Patrzymy gdzie wskoczyły nowe (zwykle `late_round_form`, `vs_opp_hand_balance`). |

## Wyniki

- Match accuracy: **60.85%** vs baseline **61.02%**
- **Delta: -0.17 p.p.** — praktycznie bez zmian, lekko gorzej
- Na targetowych slice'ach POPRAWA (np. L-vs-R × tourney_level=500: +8.6 p.p.)
- Ale na innych slice'ach POGORSZENIE (np. tourney_level=F: -20 p.p.)
- Netto się równoważy

**Lekcja**: szerokie strzelanie nie działa. Trzeba iść głęboko w jeden temat (patrz bestof5_v1 albo qfserve_v3).

## Co odpowiedzieć gdy promotor zapyta…

**Q: „Czemu ten wariant daje gorszy wynik (-0.17 p.p.) niż baseline?"**
A: Bo dodaliśmy 20 cech rozproszonych po 3 problemach. Na każdym pojedynczym problemie pomogły, ale RF musiał uczyć się 72 cech zamiast 40 — gubi sygnał w szumie. Lekcja: ten wariant uczy nas, że *focused approach* (Bo5 albo QF z dużą głębią) działa lepiej niż shotgun.

**Q: „Skoro wynik gorszy, czemu w ogóle pokazujesz ten wariant?"**
A: Bo to ważna lekcja metodologiczna. Pokazaliśmy, że naiwne „dodaj cechy do wszystkiego naraz" nie działa. Potem zrobiliśmy dwa wzorcowe focused warianty (Bo5, QFServe), które dały +2.20-2.37 p.p. To jest część historii badawczej.

**Q: „Co to znaczy 'shotgun approach'?"**
A: Analogia do strzelania śrutem — rozrzucone strzały po wielu celach naraz. W ML znaczy „dodaj wszystko co może pomóc i zobacz". Często gorsze niż focused approach (sniper — jeden cel z głębią).

**Q: „Co to jest 'expanding window' przy liczeniu cech?"**
A: Dla meczu nr 1500 jego forma liczy się z meczów 0-1499 (wszystkich PRZED nim). Nie wolno użyć meczów 1501+ bo to leakage z przyszłości. Bez tego model „wiedziałby" o przyszłych rezultatach swojego gracza i fake'owo polepszał accuracy.

**Q: „Czemu fallback do ogólnej formy a nie do 0.5?"**
A: Bo gracz może NIGDY nie grać Bo5 (młody zawodnik), ale i tak mamy o nim dużo informacji z Bo3. Fallback do ogólnej formy oddaje to: „nie znam jego Bo5 formy, ale wiem że ogólnie wygrywa 70%". 0.5 zacierałby tę informację.

**Q: „Co robi PlayerHistoryIndex i czemu jest potrzebny?"**
A: Optymalizacja wydajności. Bez niego dla każdego meczu (~3000) i każdej cechy (~16) szukamy historii gracza przez przejrzenie całej historii (18000 wierszy). To setki milionów operacji. Index trzyma mapę „gracz → indeksy meczów" i robi bisect — O(log n) zamiast O(n).

**Q: „Czemu nie tuningujesz hiperparametrów RF dla tego wariantu?"**
A: Bo chcemy uczciwego porównania (ablation study). Jeśli zmienię hiperparametry i dodam cechy, nie wiem czy +/- 0.5 p.p. pochodzi z cech czy z hiperparametrów. Trzymając hiperparametry stałe izoluję efekt cech.

**Q: „Co to są feature importances i co z nich wynika?"**
A: RF mówi „jak bardzo każda cecha wpływa na decyzje drzew" (Gini importance). Sortujemy malejąco. Dla sliceaware top 5 to zwykle: `rank_diff`, `p1_rank_log`, `p2_rank_log`, `form_diff`, `late_round_form`. Trzy pierwsze to klasyki, ostatnia to nasza nowa cecha — czyli RF JĄ używa, mimo że ogólne accuracy nie wzrosło.
