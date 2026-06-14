# Czyszczenie danych i kolumn przy nowych sezonach

Notatka odpowiadająca na uwagę promotora: *„czy kolumny oraz dane są czyszczone
na nowe lata"*. Opisuje, jak pipeline przygotowuje dane przy dokładaniu kolejnego
sezonu (2026, 2027…), co było ryzykiem i co zostało zrobione.

## O co chodziło w uwadze

Pytanie dotyczy **odporności i powtarzalności przygotowania danych**, gdy do
projektu dochodzi nowy rok. Konkretnie: czy czyszczenie (usuwanie braków,
kodowanie kategorii, obsługa kolumn) zadziała tak samo i poprawnie dla sezonu,
którego model wcześniej nie widział — czy też zależy od tego, jakie lata akurat
są wczytane (co czyniłoby wyniki nie w pełni reprodukowalnymi).

## Co było ryzykiem (zweryfikowane na danych)

1. **Niestabilne kodowanie kategorii.** `LabelEncoder` był dopasowywany od nowa
   przy każdym uruchomieniu na aktualnie wczytanych danych. Ponieważ nawierzchnia
   `Carpet` występuje tylko w starszych sezonach (do ~2017), a `tourney_level='O'`
   (igrzyska) tylko w 2021 i 2024, liczba przypisana np. „Hard" mogła się
   **przesuwać** zależnie od składu historii. Co gorsza, zapisany enkoder
   **rzuciłby wyjątek** na nowym sezonie z nieznaną kategorią.

2. **Ciche grupowanie nieznanych rund.** Słownik `ROUND_ORDER` nie zawierał
   kodów `ER` (early rounds) ani `3rd/4th`, więc `.fillna(3)` przypisywał im
   wartość 3 (=poziom R32) bez żadnego ostrzeżenia. Nowy kod rundy w przyszłym
   sezonie zachowałby się tak samo — zostałby po cichu zmielony.

3. **Niejednolite kolumny między plikami.** Kolumna `indoor` istnieje tylko
   w plikach 2018–2024. Baseline jej nie używa, więc nie pęka, ale potwierdza,
   że roczne CSV-ki nie mają identycznego zestawu kolumn.

## Co zostało zrobione (`src/tennis_model.py`)

Jeden, jawny tor czyszczenia dla **każdego** sezonu — obecnego i przyszłego:

- **`clean_match_data(df, źródło, require_full=)`** — waliduje, że plik ma
  wszystkie wymagane kolumny (`cols_base`); brak kolumny w danych docelowych to
  czytelny `KeyError` (zamiast kryptycznego błędu pandas), w historii tylko
  ostrzeżenie. Raportuje, ile wierszy usunął `dropna` i z jakiego powodu
  (głównie mecze bez statystyk serwisowych: walkowery, kreczy).

- **Stałe słowniki kodowania** `SURFACE_ENCODING`, `TOURNEY_LEVEL_ENCODING`,
  `ROUND_ORDER` — kody są **identyczne niezależnie od wczytanych lat**
  („Hard" to zawsze 3, nawet gdy w danych nie ma Carpet). Wartości odpowiadają
  dotychczasowej kolejności alfabetycznej `LabelEncoder`, więc **wyniki
  kanoniczne pozostają bez zmian** (baseline 2025: match 0.6566, identycznie).

- **Jawny kod + ostrzeżenie dla nieznanej kategorii.** Nieznana nawierzchnia lub
  poziom dostaje kod `-1` (`UNKNOWN_CATEGORY_CODE`), nieznana runda — `ROUND_DEFAULT=3`,
  a `encode_categories` **wypisuje `UWAGA`** z nazwą i liczbą wierszy. Zamiast
  cichego przesunięcia lub crasha mamy widoczny sygnał „dodaj tę kategorię do
  słownika, jeśli ma być rozróżniana".

- **`ROUND_ORDER` uzupełniony** o `ER` i `3rd/4th` (jawne wartości zamiast
  cichego fallbacku).

## Dowód, że to działa (testy)

| Scenariusz | Zachowanie |
|---|---|
| Nieznana nawierzchnia `PlexiPave` (sezon 2027) | kod `-1` + `UWAGA` |
| Nieznany poziom `XYZ` | kod `-1` + `UWAGA` |
| Nieznany kod rundy `NewRound` | kod `3` + `UWAGA` |
| Brak wymaganej kolumny w pliku | czytelny `KeyError` |
| Dane bez Carpet | „Hard" nadal = 3 (kodowanie stabilne) |
| Pełny baseline 2025 | match **0.6566** — bez zmian względem wcześniejszych wyników |

## Jak dodać nowy sezon (np. 2026)

1. Wrzuć `data/sample_data/atp_matches_2026.csv` (format Jeff Sackmann).
2. Uruchom z `TENNIS_TARGET_YEAR=2026` (historia od `TENNIS_HISTORY_START`,
   domyślnie 2001). Żadnej edycji kodu — lata są sterowane zmiennymi środowiskowymi.
3. Sprawdź log: jeśli pojawi się `UWAGA` o nieznanej kategorii, zdecyduj, czy
   dodać ją do odpowiedniego słownika (`SURFACE_ENCODING` / `TOURNEY_LEVEL_ENCODING`
   / `ROUND_ORDER`), czy zostawić jako `-1`/`3`.
4. Kursy: `python src/fetch_odds.py --years 2026` + `python src/fetch_odds_betexplorer.py --years 2026`
   (pipeline kursowy normalizuje schemat — nazwy, daty ISO, typy — przy każdym zapisie).

## Co już było dobre (warto powiedzieć na obronie)

- Lata są w pełni parametryzowane (`TENNIS_TARGET_YEAR`, `TENNIS_HISTORY_START`),
  `data_file()` sam generuje ścieżkę — dodanie roku to zmienna środowiskowa,
  nie edycja kodu.
- `dropna` czyści braki automatycznie i identycznie dla każdego sezonu.
- Pipeline kursowy (`fetch_odds*.py`) już wcześniej normalizował schemat na nowe
  lata (migracja nazw kolumn, typy `Int64`, daty ISO, `--recompute-averages`).
