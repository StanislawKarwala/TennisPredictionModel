# Plan rozwoju modelu — jak podnieść match_accuracy

> Dokument powstał z analizy wieloagentowej (53 agentów Opus 4.8: 6× analiza kodu, 5× research internetowy, weryfikacja adversarialna każdego kluczowego znaleziska i linku). Wszystkie pozycje mają etykietę **[ZWERYFIKOWANE]** (agent ponownie przeczytał kod i potwierdził) lub **[do potwierdzenia]**.

## TL;DR — kolejność działań wg zwrotu z wysiłku

| # | Działanie | Szac. zysk | Wysiłek | Status |
|---|---|---|---|---|
| 1 | Napraw bug `calculate_tournament_path_stats` (cecha path liczona na całej karierze) | +0.2-0.5 p.p. + poprawność | XS | ✅ zweryfikowane |
| 2 | Napraw metrykę `match_accuracy` (liczona jednostronnie) | poprawność pomiaru | S | ✅ zweryfikowane |
| 3 | Limit czasowy na `tail(10)` (mecz sprzed 6 lat = wczorajszy) | +0.5-1.5 p.p. | S | ✅ zweryfikowane |
| 4 | **HistGradientBoosting zamiast/obok RF** (już w sklearn) | **+1.5-3 p.p.** | M | ⚠️ partially |
| 5 | EWMA zamiast SMA (recency weighting) | +1-2 p.p. | M | ✅ zweryfikowane |
| 6 | **Surface speed index + interakcja serve×speed** | +1-2 p.p. | M-L | patrz sekcja |
| 7 | Cechy zmęczenia (rest days + minuty w turnieju) | +0.5-1.5 p.p. | S-M | ⚠️ partially |
| 8 | scoring CV `log_loss`/`roc_auc` zamiast `accuracy` | +0.3-0.8 p.p. | XS | ✅ zweryfikowane |
| 9 | Ensemble/stacking baseline+bestof5+qfserve | +0.5-1.5 p.p. | M | ⚠️ partially |
| 10 | Walk-forward test na wielu latach (wiarygodność CI) | CI z ±4 → ±1 p.p. | M | ✅ zweryfikowane |

**Najważniejsza rekomendacja:** największy pojedynczy zysk to **#4 (gradient boosting)** — bez nowych zależności (`HistGradientBoostingClassifier` jest w sklearn, którego już używasz). Ale **najpierw napraw #1-3** (bugi), bo inaczej będziesz mierzyć poprawę na zepsutej metryce.

---

## CZĘŚĆ A — Zweryfikowane błędy (napraw NAJPIERW)

### A1. `calculate_tournament_path_stats` liczy cechę na całej karierze, nie na turnieju ✅
**Plik:** `src/tennis_model_sliceaware_qfserve_v3.py:558-576` + `get_player_history:269-275`
**Problem:** Funkcja ma filtrować mecze tego samego `tourney_id` (droga gracza w bieżącym turnieju), ale ponieważ `get_player_history` używa globalnego `_HISTORY_INDEX`, filtr `same_tournament` jest cicho ignorowany — cechy `tourney_path_opp_strength` i `tourney_path_match_count` liczą się na CAŁEJ historii kariery. To nie leakage (dane są z przeszłości), ale **cecha nie niesie sygnału, który miała nieść**.
**Fix:** Wewnątrz `calculate_tournament_path_stats` filtrować jawnie po `current_row["tourney_id"]` na przekazanych `past_matches` (nie przez globalny index). Zweryfikowano: poprawienie sprawi, że cechy path zaczną mierzyć realną „trudność drogi".
**Zysk:** +0.2-0.5 p.p. (cechy `seed_context_diff`/`tourney_path_*` są w top-30 importance, więc realnie używane).

### A2. `match_accuracy` liczona tylko z perspektywy zwycięzcy ✅
**Plik:** `src/tennis_model.py:865-871` + `apply_match_level_threshold:1087-1093`
**Problem:** Metryka bierze tylko wiersze `y==1` (gdzie p1 = faktyczny zwycięzca) i liczy `mean(p1_prob > 0.5)`. Ignoruje lustrzany wiersz `y==0` tego samego meczu. Model może być niespójny (dla zwycięzcy dać 0.55, dla lustra 0.60 dla przeciwnika — dwie sprzeczne predykcje). To dlatego threshold tuning dawał absurdalne 93%.
**Fix:** Połączyć obie perspektywy po `match_id`: `P_winner = (proba_y1 + (1 - proba_y0)) / 2`, potem `correct = P_winner > 0.5`. Metryka symetryczna, odporna na arbitralny labeling.
**Skutek:** Pomiar realniejszy (może spaść 0.5-1.5 p.p.), ale **poprawny i wiarygodny do pracy magisterskiej**.

### A3. `tail(10)` bez ograniczenia czasowego ✅
**Plik:** `src/tennis_model.py` — `calculate_form`, `calculate_serve_stats`, `_serve_stats_from_player_history`
**Problem:** Forma i serwis biorą ostatnie 10 meczów gracza, ale bez limitu czasu. Dla gracza grającego rzadko (kontuzja, powrót) „ostatnie 10 meczów" może sięgać 3-6 lat wstecz — mecz sprzed 6 lat traktowany jak wczorajszy.
**Fix:** Dodać okno czasowe (np. tylko mecze z ostatnich 365 dni) ALBO recency weighting (→ patrz D1, EWMA rozwiązuje to elegancko).
**Zysk:** +0.5-1.5 p.p.

### A4. Mniejsze (low/medium) — warte uwagi przy refaktorze
- `dropna(cols_base)` usuwa mecze bez statystyk serwisowych (walkowery) — tracisz historię. Rozważ `dropna` tylko na kolumnach krytycznych. (+0.3-1 p.p.)
- Threshold tuning jest **martwym kodem** (zawsze zwraca 0.5) — `THRESHOLD_GRID`, `select_match_level_threshold` do usunięcia.
- `surface` kodowane ordinalnie `LabelEncoder` (Clay<Grass<Hard alfabetycznie) — RF sobie radzi, ale to przypadkowy porządek. One-hot albo natywne kategorie (HistGBDT je obsługuje!).
- `is_lefty_matchup` łapie też rękę `U`/NaN — myląca nazwa, drobny błąd semantyczny.

---

## CZĘŚĆ B — Surface Speed Index (Twoja główna prośba)

### Werdykt z researchu: są DWIE dobre drogi

#### Droga 1 (REKOMENDOWANA): policz własny proxy z danych, które JUŻ masz
**Dlaczego:** zero zewnętrznych zależności, pełna kontrola nad leakage, te same klucze co Twoje dane.
**Metodologia** (potwierdzona przez Tennis Abstract „The Speed of Every Surface" i Data For Tennis):
```
court_pace_index(turniej) = adjusted ace rate
  raw_pace = 0.5 * (suma_ace / suma_svpt) 
           + 0.5 * ((w_1stWon + l_1stWon) / (w_1stIn + l_1stIn))
  → agregacja per (tourney_base, surface), znormalizowana do średniej tour
```
**Krytyczne zasady anty-leakage (z weryfikacji):**
1. Liczyć CPI **wyłącznie z historii 2018-2023** (rozłącznie z ocenianym 2024).
2. Normalizację (z-score/min-max) **dopasować tylko na historii**, potem zastosować do 2024 (nie liczyć skali z testu!).
3. Klucz agregacji = `tourney_id` bez prefiksu roku, ale **z minimalnym wsparciem ≥20-30 meczów**. Klucze poniżej progu (Davis Cup — `M-DC-2024-...` NIE agreguje się po usunięciu roku!, drobne eventy) → fallback do średniej danej nawierzchni.

#### Droga 2: pobierz gotowe dane z Tennis Abstract
**Najlepsze źródło — zweryfikowane jako działające:**
- **`https://www.tennisabstract.com/cgi-bin/surface-speed.cgi?year=2024`** (parametr `year=` od 1991 do 2026)
- Per turniej × nawierzchnia × rok. **Liczone z TYCH SAMYCH danych Jeff Sackmann** → join po `(tourney_name, year, surface)` działa bezpośrednio.
- ⚠️ To tabela HTML (nie CSV) — trzeba sparsować (pandas `read_html` albo BeautifulSoup), iterując po latach.
- Statyczny raport: `https://www.tennisabstract.com/reports/atp_surface_speed.html`

**Źródła odrzucone/słabe:** courtspeed.com (dane ładowane JS-em, brak endpointu), tennisedge.io (~14 turniejów, do przepisania ręcznie), Kaggle (brak — to te same dane Sackmanna), tennis-data.co.uk (tylko nazwa nawierzchni, ale ma kursy bukmacherskie — patrz E).

### Jak wpiąć (zweryfikowany punkt integracji)
1. **Darmowy pierwszy krok:** kolumna **`indoor`** (O/I/NaN) **JUŻ JEST w 2024.csv** (2422 outdoor / 408 indoor / 246 NaN) i **nigdzie nieużywana**. Dodaj ją do `EXTRA_CONTEXT_COLUMNS` — natychmiastowa darmowa cecha (korty indoor są szybsze).
2. Nową kolumnę `court_pace_index` dodać do `EXTRA_CONTEXT_COLUMNS` (`src/tennis_model_sliceaware_qfserve_v3.py:77`) i `load_context_frame`.
3. ⚠️ **NIE wpinać przez `symmetrize_data` ani `SYMMETRIC_FEATURE_SPECS`** — to by się wysypało `KeyError` (mechanizm wymaga kolumn `w_*`/`l_*` w raw_data). Wpiąć przez `attach_targeted_features` (merge po `match_id`).
4. **Kluczowa cecha — interakcja serve × speed** (główne źródło zysku):
   ```
   serve_advantage_diff = p1_serve_strength - p2_serve_strength   # z compose_serve_score
   serve_x_speed_diff = serve_advantage_diff * court_pace_index
   ```
   `court_pace_index` jest symetryczny (kontekst meczu, jeden na wiersz), więc `serve_x_speed_diff` poprawnie zmienia znak przy symetryzacji. Liczyć WEWNĄTRZ `attach_targeted_features` z już-symetryzowanych `p1_*`/`p2_*` (nie przez SYMMETRIC_FEATURE_SPECS).
5. ⚠️ `surface_encoded` (int z LabelEncoder) **NIE nadaje się** jako mnożnik prędkości — to przypadkowy kod kategoryczny. Trzeba osobnego mapowania `surface → pace` albo policzonego CPI.

**Zysk:** sama cecha CPI: +0.3-0.8 p.p.; interakcja serve×speed: +0.7-1.5 p.p. (to ona jest wartościowa).

---

## CZĘŚĆ C — Modelowanie (największy zwrot)

### C1. HistGradientBoosting / XGBoost zamiast Random Forest ⚠️ NAJWIĘKSZY POJEDYNCZY ZYSK
**Plik:** `src/tennis_model.py:780`
Gradient boosting prawie zawsze przebija RF na danych tabelarycznych. **`HistGradientBoostingClassifier` jest już w sklearn** (zero nowych zależności), ma natywną obsługę kategorii (surface, tourney_level bez encodingu!) i braków danych.
**Plan:** dodać jako alternatywny estymator, dostroić `learning_rate`, `max_iter`, `max_leaf_nodes`, `l2_regularization`. Porównać uczciwie na tej samej walidacji.
**Zysk:** +1.5-3 p.p. + lepsza kalibracja.

### C2. scoring CV: `log_loss`/`roc_auc` zamiast `accuracy` ✅
**Plik:** `src/tennis_model.py` (RandomizedSearchCV)
`accuracy` jest progowa i szumowa przy doborze hiperparametrów. Dla zadania probabilistycznego `neg_log_loss` lub `roc_auc` daje stabilniejszy wybór. **Najtańsza zmiana** (jeden parametr).
**Zysk:** +0.3-0.8 p.p. + stabilniejszy wybór HP.

### C3. Ensemble / stacking trzech modeli ⚠️
Połączyć `baseline` + `bestof5_v1` + `qfserve_v3`. Dwie opcje:
- **Routing** (proste): Bo5 → bestof5_v1, R128×L-vs-R → qfserve_v3, reszta → bestof5_v1.
- **Stacking** (lepsze): meta-model na out-of-fold predykcjach trzech modeli.
**Zysk:** +0.5-1.5 p.p. (najwięcej na best_of=5).

### C4. Walk-forward test na wielu latach ✅
**Problem:** 590 meczów testowych (1 rok) → CI accuracy ~±4 p.p. Różnice między wariantami (0.17-2.37 p.p.) częściowo toną w szumie.
**Fix:** rolling/walk-forward — trenuj na 2018-2022, testuj 2023; trenuj 2018-2023, testuj 2024; itd. Uśrednij. CI spada do ~±1 p.p. → móżesz **wiarygodnie** twierdzić, że wariant X jest lepszy.

### C5. Pozostałe (medium)
- Kalibracja: porównać `isotonic` vs `sigmoid` + kalibracja CV zamiast na 470 próbkach val.
- Jawne `class_weight`, lepsza obsługa braków (HistGBDT robi to natywnie).
- Finalny model trenowany na `shuffle=True` po CV — rozjazd z danymi CV; rozważyć spójność.

---

## CZĘŚĆ D — Nowe cechy z danych, które JUŻ masz (bez zewnętrznych źródeł)

| Cecha | Skąd policzyć | Zysk | Status |
|---|---|---|---|
| **D1. EWMA zamiast SMA** | recency weighting formy/serwisu (α≈0.18 krótki, 0.05 długi); jest gotowy wzorzec w `experiments_archive/tennis_model_ewma.py` | +1-2 p.p. | ✅ |
| **D2. Rest days** | `tourney_date` − data ostatniego meczu gracza | +0.5-1.5 p.p. | ⚠️ |
| **D3. Minuty w turnieju** | suma `minutes` z wcześniejszych rund tego turnieju (zmęczenie) | +0.5-1 p.p. | ⚠️ |
| **D4. H2H na nawierzchni** | bilans bezpośredni filtrowany po `surface` + recency H2H | +0.3-0.8 p.p. | ✅ |
| **D5. Tiebreak record** | parsing kolumny `score` (clutch w końcówkach setów) | +0.3-0.7 p.p. | ✅ |
| **D6. Break-point conversion (return)** | brakująca połowa gry BP: ile BP gracz wykorzystał na returnie | +0.5-1 p.p. | ⚠️ |
| **D7. Momentum** | deciding-set record, win streak | +0.3-0.8 p.p. | ⚠️ |
| **D8. Indoor/outdoor** | kolumna `indoor` już w CSV, nieużywana | drobny | ✅ |

**Marnowane kolumny w danych:** `minutes`, `score`, `indoor` — wystarczają do D2/D3/D5/D8 bez żadnych zewnętrznych danych.

---

## CZĘŚĆ E — Przyspieszenie kodu

| # | Co | Zysk | Status |
|---|---|---|---|
| E1 | `symmetrize_data` na `iterrows`+dict per wiersz → wektoryzacja (wołane 4× na przebieg) | 5-15× szybsza symetryzacja | ✅ |
| E2 | `calculate_context_serve_profile`/`extract_player_match_serve_metrics` na `iterrows` → wektoryzacja | 4-8× szybsze feature engineering w wariantach | ⚠️ |
| E3 | Martwy kod: stare `calculate_form`/`get_h2h`/`calculate_serve_stats` (niewektoryzowane) do usunięcia | czystość | ✅ |
| E4 | Redundantne `.copy()` całych ramek dla nadpisania 1 kolumny | drobny | low |
| — | **`n_jobs` jest już poprawnie ustawione (`-1`) wszędzie — NIE bug** | — | ✅ |

---

## E (bonus) — Dodatkowe dane zewnętrzne warte rozważenia

- **Kursy bukmacherskie** — `tennis-data.co.uk` (ZIP roczne, ATP od 2000, kursy od 2001; działa po **HTTP**, nie HTTPS). Implied probability z kursów bije większość modeli akademickich; świetny benchmark i potężna cecha (uwaga: część recenzentów traktuje jako leakage).
- **Surface-adjusted Elo** (Angelini EJOR, Sipko Imperial 2015) — dwa Elo (ogólne + per-surface), ważona średnia. Liczalne z samych danych Sackmanna. Udokumentowany zysk w literaturze.

---

## Sugerowana ścieżka (sprint po sprincie)

**Sprint 1 (poprawność — 1 dzień):** A1 + A2 + A3 + C2. Napraw bugi i metrykę, zmień scoring. Teraz mierzysz poprawnie.
**Sprint 2 (model — 2-3 dni):** C1 (HistGradientBoosting). Prawdopodobnie największy skok.
**Sprint 3 (cechy — 2-3 dni):** D1 (EWMA) + D2/D3 (zmęczenie) + B (surface speed: indoor → proxy CPI → interakcja serve×speed).
**Sprint 4 (wiarygodność — 1-2 dni):** C4 (walk-forward) + C3 (ensemble). Udowodnij zysk statystycznie.
