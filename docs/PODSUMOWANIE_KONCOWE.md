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

## Pliki eksperymentów (zostają jako dowód, NIE importowane do main)
`main_48_cech_hgb.py` (Sprint 2), `main_48_cech_surface_speed.py` / `main_48_cech_fatigue.py` / `main_48_cech_enriched.py` / `main_48_cech_ewma_ablation.py` (Sprint 3), `main_48_cech_walkforward.py` / `main_48_cech_salvage.py` (Sprint 4).
