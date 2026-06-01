# Podsumowanie końcowe — cykl rozwoju modelu (Sprinty 1-4)

Dokument domykający. Szczegóły liczbowe: `WYNIKI_SPRINTOW.md`. Pojęcia: `SLOWNICZEK_POJEC.md`.

## Co zrobiliśmy i czego się dowiedzieliśmy

Przeszliśmy pełny, rygorystyczny cykl: od naprawy poprawności, przez zmianę algorytmu i nowe cechy, po wieloletnią walidację. **Najcenniejszy wynik jest metodologiczny, nie accuracy.**

### 1. Sprint 1 — poprawki poprawności (SCALONE do `main_48_cech.py`)
Cztery zmiany, które bronią się niezależnie od accuracy:
- **A1**: naprawiony bug cechy „drogi w turnieju" (liczona na całej karierze → bieżący turniej).
- **A2**: metryka match_accuracy z jednostronnej → symetrycznej (uśrednia obie perspektywy). Stara dawała niespójności (threshold tuning → fałszywe 93%).
- **A3**: okno 365 dni na formę/serwis (koniec traktowania meczu sprzed lat jak wczorajszego).
- **C2**: dobór hiperparametrów po `neg_log_loss` (właściwe kryterium probabilistyczne).

### 2. Sprint 2 — HistGradientBoosting: wynik NEGATYWNY
Gradient boosting (z natywnymi kategoriami włącznie) NIE pobił RF. Hipoteza „+1.5-3 p.p." z literatury nie obowiązuje na ~3500 próbkach. **RF zostaje.**

### 3. Sprint 3 — nowe cechy: pozorny zysk
surface_speed (+1.69), fatigue (+1.36), zbiorczy (+2.03 p.p.) na pojedynczym teście 2024 — wyglądało przekonująco (val/test/match spójnie w górę, lepszy Brier). EWMA słaby/niespójny.

### 4. Sprint 4 — walk-forward: pozorny zysk OBALONY
Walidacja przez 4 sezony (2021-2024, 2220 meczów): **pooled delta +0.09 p.p., McNemar p=0.93.** Delta dodatnia tylko 2/4 lat. Nawet wąskie warianty (2 cechy) — p=0.37, brak istotności. **Cechy nie generalizują.**

## Trzy twarde wnioski

1. **Single test set kłamie.** +2.03 p.p. na 2024 było szumem (CI ±4 p.p. przy 590 meczach). Bez walk-forward wpisalibyśmy do pracy fałszywy zysk. To główna lekcja metodologiczna.

2. **Model bije ranking tylko o +0.95 p.p.** RF 64.73% vs naiwny „wygrywa wyżej notowany" 63.78% (pooled). W 2024 RF był *gorszy* od rankingu. Większość accuracy to „faworyci wygrywają", nie feature engineering. Zmienność 61-67% rok-do-roku = skład meczów (faworyci/upsety), nie cechy.

3. **Jesteśmy na suficie feature-based.** Literatura: modele cechowe 64-67%, sufit bez kursów ~70% (Elo), z kursami ~72-76%. Nasz p=0.93 dla nowych cech = konsensus, że dalsze cechy serwis/forma to ślepa uliczka.

## Co robić dalej (jedyne realne dźwignie)

Wg literatury i naszych testów ponad ranking podnoszą TYLKO dwie rzeczy:

1. **Surface-adjusted Elo** zamiast/obok surowego rankingu. Tennis Abstract publikuje gotowe ratingi Elo per nawierzchnia (te same dane Sackmanna). To prawdopodobnie jedyna „cecha", która da robust +kilka p.p. (do ~70%).

2. **Kursy bukmacherskie** (implied probability) jako cecha I/lub benchmark. `tennis-data.co.uk` (ZIP roczne, działa po HTTP). Podnoszą do ~72-76%, ale: (a) część recenzentów traktuje jako leakage, (b) bicie kursów jest bardzo trudne (modele ML i bukmacherzy statystycznie nieodróżnialni).

**Czego NIE robić:** kolejnych cech serwisowo-formowych — to udowodniona ślepa uliczka (Sprint 3-4 + literatura).

## Rekomendacje raportowania (do pracy)

- Zawsze raportuj accuracy **obok naiwnego baseline „higher-rank-wins"** na tej samej próbie — pokazuje realną wartość dodaną modelu.
- Raportuj **walk-forward (wiele lat)**, nie pojedynczy test set — i podawaj CI/McNemar.
- Raportuj **log-loss/Brier/ECE** obok accuracy (jakość prawdopodobieństw, kluczowa przy ewentualnym bettingu).
- Nie porównuj pooled sezonowego (~65%) z liczbami 75%+ z blogów (to pojedyncze Grand Slamy, więcej faworytów).

## Czy więcej danych (od 2000) uzasadni powrót do XGBoost/HGB? TAK

HGB przegrał (Sprint 2), bo miał ~3540 próbek treningowych i wybrał mocno regularyzowane HP — za mało danych, by boosting rozwinął przewagę. Dane od 2000 (~72 000 meczów → ~144 000 wierszy po symetryzacji, ~20-40×) to reżim, w którym boosting zwykle bije RF. **Powrót do testu RF vs HGB vs XGBoost byłby uzasadniony.**

Zastrzeżenia: (1) dryf rozkładu — stare mecze (inne korty/technologia) mogą słabiej przewidywać; rozwiązanie: ważenie czasowe + walk-forward sprawdzający czy stare dane pomagają. (2) Sufit ~70% wciąż obowiązuje — więcej danych pomoże *osiągnąć* ~70%, nie przebić. (3) Więcej danych pomaga WSZYSTKIEMU: kurczy CI walk-forward (25 sezonów zamiast 4 → wykrywalne efekty ±0.5 p.p.), rozgrzewa Elo, daje próbki rzadkim matchupom.

**Kolejność: (1) pobierz pełne dane Sackmanna (github.com/JeffSackmann/tennis_atp, od 1968, darmowe), (2) powtórz RF vs boosting na dużym zbiorze, (3) re-tuning HP obowiązkowy.**

## Surface-adjusted Elo (Sprint 5) — najlepszy kierunek, ale jeszcze nieistotny na tych danych
Zaimplementowane w `src/main_48_cech_elo.py` z walidacją walk-forward (4 sezony). Elo liczony samodzielnie z danych Sackmanna (leakage-safe), ogólny + per-nawierzchnia, K dynamiczny (538).

| Rok | baseline | +elo | delta |
|---|---|---|---|
| 2021 | 0.6724 | 0.6858 | +0.0134 |
| 2022 | 0.6709 | 0.6673 | −0.0037 |
| 2023 | 0.6399 | 0.6275 | −0.0125 |
| 2024 | 0.6102 | 0.6305 | +0.0203 |
| **POOLED** | **0.6473** | **0.6518** | **+0.0045** |

McNemar p=0.50 (nieistotne), dodatnie 2/4 lat. **ALE cechy Elo dominują ważność**: `elo_diff` średni rank 2.0/44, `elo_win_prob` rank 3.5.

**Kluczowa interpretacja:** Elo to genuinie silny sygnał (top-2 cecha), ale **redundantny z rankingiem, który baseline już ma** (elo_diff ≈ rank_diff koncepcyjnie). Literaturowe „Elo → ~70%" dotyczy Elo jako GŁÓWNEGO sygnału vs sam ranking — nie Elo DODANEGO do modelu z 40 cechami zawierającymi już ranking. Nasz pooled +0.45 p.p. (najlepszy additive wynik cyklu: surface_speed +0.09, fatigue ~0, Elo +0.45) jest w dobrym kierunku, ale w granicach szumu przy 2220 meczach.

**Kiedy Elo zapłaci:** (1) **więcej danych historycznych** — Elo rozgrzewa się przez lata; z danymi od 2000 ratingi byłyby dużo dokładniejsze (14+ lat warmup vs 3-6) i delta mogłaby przekroczyć istotność; (2) jako element modelu MINIMALNEGO (Elo zamiast rankingu) — nie dodatek. To bezpośrednio łączy się z rekomendacją „więcej danych".

**Powtarzający się wzorzec (ważny):** KAŻDA dobra cecha (surface_speed, Elo) daje 2024 dokładnie +0.0203 (oba lądują na 0.6305), a szkodzi w 2022/2023. Bo baseline RF w 2024 był słaby (61.02% < naiwny ranking 61.36%) — każdy rank-podobny sygnał „naprawia" tam ~2 p.p. To regresja do średniej, nie efekt cechy. Potwierdzone na dwóch niezależnych zestawach cech.

## Migracja danych: pełny zbiór ATP 2001-2026

Dane rozszerzone z 2018-2024 do **2001-2026**, format `atp_matches_{rok}.csv` (standard Jeff Sackmann). Prefiks `atp_` chroni przed pomyleniem z WTA w przyszłości (przełącznik `TENNIS_TOUR`).

**Zmiany w kodzie (wszystkie pliki):**
- Nazwy plików: `{rok}.csv` → `atp_matches_{rok}.csv` (helper `data_file(year)`).
- **Rok docelowy domyślnie 2025** (pełny sezon; 2026 to pół roku, 67 turniejów — nie wszystkie nawierzchnie). Env `TENNIS_TARGET_YEAR`.
- **Historia od 2001** (env `TENNIS_HISTORY_START`).
- Usunięto `is_indoor` (standardowe pliki Sackmanna nie mają tej kolumny; cecha i tak była bezużyteczna rank 44/44).
- Walk-forward/elo/salvage: domyślnie 6 sezonów testowych [2020-2025] (env `TENNIS_WF_YEARS`).

**Efekt na baseline (target 2025, historia 2001-2024):** match accuracy **65.66%** (vs 61.02% na 2024), Brier 0.2174 (lepszy). Wzrost z bogatszej historii (lepsze cechy formy/H2H/serwisu) + 2025 jako rocznik. surface_speed na 2025: delta +0.0000 (potwierdza brak robust sygnału).

### ⚠️ KLUCZOWE: architektura trenuje TYLKO na roku docelowym
Historia (2001-2024) służy do liczenia CECH dynamicznych, ale model trenuje się na 60% roku docelowego (~1770 meczów → ~3500 próbek). Czyli:
- Więcej historii → lepsze cechy (stąd wzrost baseline), ALE
- **zbiór treningowy boostingu nadal ~3500 próbek** → samo podpięcie danych 2001+ NIE rehabilituje XGBoost/HGB.
- Żeby boosting dostał szansę: trzeba **wielo-sezonowego treningu** (trenuj 2001-2023 ≈ 130k próbek, waliduj 2024, testuj 2025). To osobna, większa zmiana architektury — TODO.

## Pliki eksperymentów (zostają jako dowód, NIE importowane do main)
`main_48_cech_hgb.py` (Sprint 2), `main_48_cech_surface_speed.py` / `main_48_cech_fatigue.py` / `main_48_cech_enriched.py` / `main_48_cech_ewma_ablation.py` (Sprint 3), `main_48_cech_walkforward.py` / `main_48_cech_salvage.py` (Sprint 4).
