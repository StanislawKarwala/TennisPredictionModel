# Opis pliku `main_48_cech_multiseason.py` — wielo-sezonowy trening + test boostingu

> Pojęcia techniczne (CV, neg_log_loss, Brier, walk-forward, symetryzacja, boosting) są wyjaśnione w `SLOWNICZEK_POJEC.md`. Wyniki liczbowe: `PODSUMOWANIE_KONCOWE.md`.

## Co ten plik robi w jednym zdaniu
Zmienia architekturę treningu z „tylko rok docelowy" (~3500 próbek) na **wielo-sezonową** (trening na wielu latach naraz, dziesiątki tysięcy próbek) i na tych dużych danych uczciwie porównuje trzy algorytmy — Random Forest vs HistGradientBoosting vs XGBoost — żeby sprawdzić, czy więcej danych w końcu daje przewagę boostingowi.

## Dlaczego ten plik w ogóle powstał

W Sprincie 2 HistGradientBoosting **nie pobił** Random Forest. Ale wtedy model trenował się tylko na jednym roku (~3500 próbek). Hipoteza: boosting (gradient boosting) potrzebuje DUŻO danych, żeby pokazać przewagę — przy małej próbce regularyzuje się do prostego modelu.

**Kluczowy problem architektury bazowej:** w `main_48_cech.py` historia (np. 2001-2024) służy WYŁĄCZNIE do liczenia cech dynamicznych (forma, H2H, serwis), a model trenuje się tylko na 60% roku docelowego. Czyli samo podpięcie więcej danych historycznych **nie powiększa zbioru treningowego** — boosting dalej dostaje ~3500 próbek.

Ten plik to naprawia: **trenuje na wszystkich meczach z wielu sezonów** (np. 2000-2023), co daje ~140 000 próbek. Dopiero to jest właściwy test hipotezy „więcej danych → boosting wygrywa".

## Główne założenia

1. **Podział po sezonie, nie po dacie** — `season` = rok PLIKU (`atp_matches_2025.csv` → sezon 2025), a NIE `tourney_date.dt.year`. Powód: sezon 2025 zaczyna się od United Cup z końca grudnia 2024, więc data myliłaby przynależność do sezonu.

2. **Trening / walidacja / test rozłączne czasowo** — trening = sezony [TRAIN_START .. VAL_YEAR−1], walidacja = VAL_YEAR (2024), test = TEST_YEAR (2025, cały sezon ~2650 meczów). Zero leakage: testujemy na sezonie, którego model nigdy nie widział.

3. **Te same 40 cech co baseline** — reużywamy funkcje feature-engineering z `main_48_cech.py` przez namespace (`add_dynamic_features`, `symmetrize_data`). Jedyne zmienne to ILOŚĆ danych treningowych i ALGORYTM — czysty ablation.

4. **Rozgrzewka cech (warmup)** — sezony przed TRAIN_START służą tylko do policzenia cech (forma, H2H) dla pierwszych meczów treningowych, ale same nie wchodzą do treningu. Gdy trenujemy od najwcześniejszego sezonu (2000), warmup jest pusty → mecze 2000 mają cold-start (forma=0.5), co jest akceptowalne (mały ułamek danych).

5. **LabelEncoder fitowany TYLKO na treningu** — kodowanie nawierzchni/poziomu turnieju uczone na danych treningowych (bez wglądu w walidację/test). Nieznane kategorie w teście mapowane bezpiecznie (fallback), więc np. brak Carpet w 2025 nie psuje niczego.

6. **Dobór hiperparametrów po neg_log_loss** — RandomizedSearchCV z TimeSeriesSplit (chronologiczny), spójnie z resztą projektu po Sprincie 1.

## Dlaczego takie wybory, a nie inne?

- **Czemu reużycie funkcji baseline przez namespace, a nie kopiowanie kodu?** Żeby cechy były DOKŁADNIE te same co w baseline. Kopiowanie groziłoby subtelnymi różnicami, które zafałszowałyby porównanie. Koszt: jednorazowe uruchomienie baseline (~5 min), żeby pobrać funkcje.
- **Czemu trening od 2000/2010, a nie tylko ostatnie 2-3 lata?** Bo cały sens to DUŻO danych dla boostingu. Im więcej sezonów treningowych, tym większa szansa, że gradient boosting pokaże przewagę (jeśli istnieje).
- **Czemu test na CAŁYM sezonie 2025, a nie 20%?** Bo ~2650 meczów daje CI ~±2 p.p. zamiast ±4 p.p. przy 590 meczach. Wiarygodniejszy pomiar (lekcja ze Sprintu 4).
- **Czemu cv=3, a nie 5, i mały n_iter?** Kompromis obliczeniowy. Trening na 140k próbkach × wiele fitów × 3 algorytmy to godziny. cv=3 i n_iter 8-12 wystarcza, by ocenić, czy boosting wygrywa (a nie do produkcyjnego strojenia).
- **Czemu osobny plik, a nie zmiana baseline?** Żeby nie burzyć działającego pipeline'u (baseline + warianty + walk-forward). To eksperyment architektoniczny, nie nowy model produkcyjny.

## Ważne metody (1:1 z kodu)

| Metoda | Co robi |
|---|---|
| `data_file(year)` | Buduje ścieżkę `{TOUR}_matches_{year}.csv` (np. `atp_matches_2010.csv`). Przełącznik `TOUR` (atp/wta). |
| `run_baseline_quietly()` | Uruchamia `main_48_cech.py` raz z wyciszonym stdout i zwraca namespace — z niego pobieramy funkcje feature-engineering i listę cech. |
| `load_years(years, cols_base)` | Wczytuje wskazane sezony, sortuje chronologicznie, dropna na cols_base, **taguje kolumną `season` = rok pliku**. Zwraca jedną połączoną ramkę. |
| `add_static_features(df, ROUND_ORDER)` | Dolicza cechy statyczne jak w baseline: log rankingu/punktów, flagi leworęczności, `round_encoded`. (Surface/level kodowane osobno, po treningowym fit.) |
| `tune_and_eval(name, estimator, param_dist, n_iter, ...)` | Dla danego algorytmu: RandomizedSearchCV (neg_log_loss, TimeSeriesSplit) na treningu, fit najlepszego na pełnym (wymieszanym) treningu, ewaluacja na walidacji i teście (val/test accuracy + symetryczna match accuracy + Brier/log-loss/ECE). Zwraca słownik wyników. |
| `main()` | Orkiestracja: baseline → wczytanie sezonów → cechy dynamiczne → label encoding (fit na train) → symetryzacja → trening 3 modeli → tabela porównawcza. |

## Ważne zmienne / konfiguracja (env)

| Zmienna | Domyślnie | Co oznacza |
|---|---|---|
| `TENNIS_WARMUP_START` | 2001 | Najwcześniejszy sezon wczytywany (rozgrzewka cech). Dla testu „od 2000" ustawiamy 2000. |
| `TENNIS_TRAIN_START` | 2010 | Pierwszy sezon TRENINGOWY. Sezony [WARMUP_START..TRAIN_START−1] to tylko rozgrzewka. „Od 2000" = 2000. |
| `TENNIS_VAL_YEAR` | 2024 | Sezon walidacyjny. |
| `TENNIS_TEST_YEAR` | 2025 | Sezon testowy (pełny). |
| `TENNIS_TOUR` | atp | Prefiks plików danych (atp/wta). |
| `HAS_XGB` | — | Flaga: czy XGBoost zainstalowany. Jeśli nie — porównanie RF vs HGB. |
| `span_feat` | — | Ramka 2010/2000-2025 z policzonymi cechami dynamicznymi + statycznymi + kodowaniem. |
| `train_cv` / `train_fit` | — | Trening symetryzowany: ordered (do chronologicznego CV) i shuffled (do finalnego fitu). |
| `results` | — | Lista słowników wyników per model (val/test match, Brier, log-loss, ECE, najlepsze HP). |

## Wyniki

### Trening 2010-2023 (~72 582 próbki) — test 2025
| model | test_match | Brier | log-loss | ECE |
|---|---|---|---|---|
| RandomForest | 0.6494 | 0.2187 | 0.6260 | 0.0163 |
| HistGradBoost | 0.6460 | 0.2195 | 0.6283 | 0.0224 |
| XGBoost | 0.6494 | 0.2179 | 0.6244 | 0.0212 |

### Trening 2000-2023 (~128 848 próbek) — test 2025
| model | test_match | Brier | log-loss | ECE |
|---|---|---|---|---|
| RandomForest | 0.6475 | 0.2181 | 0.6247 | 0.0170 |
| HistGradBoost | 0.6471 | 0.2172 | 0.6226 | 0.0210 |
| XGBoost | 0.6460 | **0.2167** | **0.6211** | 0.0207 |

DELTA accuracy vs RF: HGB −0.04 p.p. (remis), XGBoost −0.15 p.p. (minimalnie gorzej).
DELTA Brier vs RF: HGB −0.0009, XGBoost −0.0015 (boosting lepiej skalibrowany).

**Wniosek:** boosting NIE pobił RF na accuracy nawet na **36× większych danych** (128k próbek). Wszystkie 3 algorytmy lądują na **~64.6-64.75%**. Ściana jest w cechach/problemie, nie w algorytmie ani ilości danych.

**Ciekawa obserwacja:** przewaga boostingu w JAKOŚCI PRAWDOPODOBIEŃSTW (Brier/log-loss) lekko rośnie z ilością danych (XGBoost Brier vs RF: −0.0008 przy 72k → −0.0015 przy 128k). Czyli więcej danych pomaga boostingowi w KALIBRACJI, ale NIE w accuracy. Ranking jakości prawdopodobieństw: XGBoost > HGB > RF. Gdyby celem był betting/ROI (jakość prawdopodobieństw), XGBoost byłby marginalnie najlepszy. Dla accuracy: wszystkie równe.

## Co odpowiedzieć gdy promotor zapyta…

**Q: „Po co osobny plik do wielo-sezonowego treningu?"**
A: Bo bazowy model trenuje się tylko na roku docelowym (~3500 próbek). Żeby uczciwie sprawdzić, czy gradient boosting (XGBoost/HGB) wygrywa na dużych danych, trzeba trenować na wielu sezonach naraz (~140k próbek). Ten plik to robi, nie ruszając działającego pipeline'u.

**Q: „Czemu boosting nie wygrał, skoro literatura mówi, że bije RF na danych tabelarycznych?"**
A: Bo sufit predykcji tenisa (~65%) jest w CECHACH i naturze problemu (upsety), nie w algorytmie. Sprawdziliśmy 3 algorytmy na zakresie danych od 3.5k do 140k próbek — wszystkie lądują na ~65%. Gdy ściana jest informacyjna, lepszy algorytm nie pomoże.

**Q: „Czy podział danych jest poprawny (brak leakage)?"**
A: Tak. Trening, walidacja i test to rozłączne sezony (np. 2000-2023 / 2024 / 2025). Podział po pliku-sezonie (nie po dacie, bo sezon 2025 zaczyna się w grudniu 2024). LabelEncoder fitowany tylko na treningu. Cechy dynamiczne liczone z expanding window (tylko mecze wcześniejsze).

**Q: „Czemu trening od 2000, skoro stare dane są mniej relevantne?"**
A: To świadomy test krańcowy — sprawdzamy, czy MAKSIMUM danych cokolwiek zmienia dla boostingu. Jeśli nawet 140k próbek nie pomaga, to ostateczny dowód, że problem nie jest w ilości danych. (W produkcji można by ograniczyć do nowszych sezonów przez `TENNIS_TRAIN_START`.)

**Q: „Czy XGBoost jest w czymś lepszy?"**
A: Tak — w jakości prawdopodobieństw (Brier/log-loss minimalnie lepsze). Gdyby celem był betting/kalibracja, XGBoost jest odrobinę lepszy. Dla samej accuracy: identycznie jak RF.
