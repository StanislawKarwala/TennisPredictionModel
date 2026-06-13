"""
Pipeline kursów bukmacherskich (tennis-data.co.uk) dla meczów ATP
=================================================================

Pobiera roczne pliki kursów z tennis-data.co.uk (Bet365, Pinnacle, Avg
— kursy zamknięcia, czyli informacja dostępna PRZED meczem), dopasowuje je do
meczów z naszych plików Sackmanna (data/sample_data/atp_matches_{rok}.csv)
i zapisuje wynik do data/odds/atp_odds_{rok}.csv.

Mecze bez znalezionych kursów są pomijane (tennis-data nie ma m.in. Davis Cup
i United Cup, a sporadycznie pojedynczych meczów) — stąd pokrycie ~85-95%.

Dopasowanie (Sackmann nie dzieli match_id z tennis-data, więc łączymy heurystycznie):
  1) klucz pary graczy: (nazwisko, pierwsza litera imienia) zwycięzcy i przegranego;
     nazwiska normalizowane (akcenty/myślniki/apostrofy -> spacja, lowercase).
     Tennis-data bywa niekonsekwentne, więc dla nazwisk Sackmanna generujemy
     warianty: wszystkie podciągi nazwiska wieloczłonowego ("Giovanni Mpetshi
     Perricard" -> "mpetshi perricard", "mpetshi", "perricard" — bo tennis-data
     ma i "Mpetshi G.", i "Mpetshi Perricard G."), wersje bez spacji ("Oconnell"
     vs "O Connell C.") oraz porządek odwrócony dla nazwisk zapisanych
     nazwisko-pierwsze ("Bu Yunchaokete" vs "Bu Y.");
  2) okno czasowe: data meczu tennis-data musi wypadać od -3 do +27 dni od
     tourney_date Sackmanna (tourney_date to data STARTU turnieju);
  3) tiebreak przy kilku kandydatach (ta sama para w oknie, np. dwa turnieje
     z rzędu): zgodność rankingów (WRank/LRank vs winner_rank/loser_rank),
     potem minimalna różnica dat; każdy wiersz tennis-data użyty najwyżej raz;
  4) fallback dla rozjazdu inicjałów (np. "Barrios M." = Marcelo Tomas Barrios
     Vera, u Sackmanna "Tomas Barrios Vera"): dopasowanie po samych nazwiskach
     pary, ale akceptowane WYŁĄCZNIE przy pełnej zgodności obu rankingów
     (rank_agreement == 2), co praktycznie wyklucza fałszywe dopasowania.

Wynikowy CSV (jeden wiersz = jeden mecz ze znalezionymi kursami):
  match_key                 -- "{tourney_id}_{match_num}" (unikalny klucz Sackmanna)
  tourney_id, match_num, tourney_date, winner_name, loser_name  -- identyfikacja
  B365_winner, B365_loser   -- kursy Bet365 na zwycięzcę / przegranego
  PS_winner,   PS_loser     -- Pinnacle
  EnglishAvg_winner, EnglishAvg_loser -- średnia rynku zagranicznego (kolumna
                               Avg z tennis-data; nazwa symetryczna do PolishAvg)
  PolishAvg_winner, PolishAvg_loser   -- średnia z polskich bukmacherów
                               (po uruchomieniu fetch_odds_betexplorer.py)
  AllAvg_winner, AllAvg_loser -- średnia ze wszystkich pojedynczych bukmacherów
  td_date, td_tournament    -- diagnostyka dopasowania
  rank_agreement            -- 2 = oba rankingi zgodne, 1 = jeden, 0 = żaden

Użycie w modelu (szkic, bez leakage -- kursy zamknięcia są pre-match):
  odds = pd.read_csv(f"data/odds/atp_odds_{TARGET_YEAR}.csv")
  df_raw["match_key"] = df_raw["tourney_id"].astype(str) + "_" + df_raw["match_num"].astype(str)
  df_raw = df_raw.merge(odds[["match_key", "EnglishAvg_winner", "EnglishAvg_loser"]], on="match_key", how="left")
  # po symetryzacji (jak elo_diff w tennis_model_elo.py):
  #   p1_odds = np.where(y == 1, EnglishAvg_winner, EnglishAvg_loser); p2_odds = odwrotnie
  #   implied_prob_diff = 1/p1_odds - 1/p2_odds
UWAGA: df[cols_base].dropna() w tennis_model.py nie zawiera tourney_id/match_num
-- do merge'a trzeba je dołożyć do wczytywanych kolumn (jak EXTRA_CONTEXT_COLUMNS
w wariantach sliceaware) albo merge'ować pozycyjnie jak w tennis_model_elo.py.

Uruchomienie:
  python src/fetch_odds.py                      # domyślnie lata 2020-2025
  python src/fetch_odds.py --years 2026
  python src/fetch_odds.py --years 2024 2025 --force-refetch
  python src/fetch_odds.py --clean --years 2020 2021

Tempo: jeden HTTP GET na rok (~0.5 MB) + dopasowanie w pandas -- sekundy na sezon
(BetExplorer/Playwright niepotrzebne dla danych historycznych).
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
ODDS_DIR = BASE_DIR / "data" / "odds"

TOUR = "atp"
DEFAULT_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Kolumny kursow w tennis-data -> nazwy wyjsciowe (sufiks _winner/_loser).
# 'EnglishAvg' = srednia rynku zagranicznego (~60 ksiazek, kolumna Avg
# w tennis-data) -- nazwa symetryczna do PolishAvg, zeby sie nie mylilo.
# 'Max' (najwyzszy kurs na rynku) celowo NIE jest pobierany: do cech modelu
# wystarczaja srednie/pojedynczy bukmacherzy; Max przydaje sie dopiero przy
# backtescie ROI zakladow (gra po najlepszej cenie) -- w razie potrzeby
# wystarczy dopisac {"Max": ("MaxW", "MaxL")} i przeliczyc rok od nowa.
ODDS_COLUMNS = {"B365": ("B365W", "B365L"), "PS": ("PSW", "PSL"),
                "EnglishAvg": ("AvgW", "AvgL")}

# Stare nazwy kolumn -> nowe (migracja plikow zapisanych przed zmiana nazwy).
LEGACY_COLUMN_RENAMES = {"Avg_winner": "EnglishAvg_winner",
                         "Avg_loser": "EnglishAvg_loser"}
# Kolumny usuwane przy normalizacji (pliki sprzed decyzji o rezygnacji z Max).
LEGACY_DROP_COLUMNS = ["Max_winner", "Max_loser"]

# Prefiksy kolumn polskich bukmacherow (pochodza z fetch_odds_betexplorer.py).
POLISH_BOOK_PREFIXES = ["Betclic", "BETFAN", "Fortuna", "Fuksiarz",
                        "LVBET", "STS", "Superbet"]
# Pojedynczy bukmacherzy do sredniej ogolnej: B365 + Pinnacle + polscy.
# 'EnglishAvg' NIE wchodzi (to juz srednia ~60 ksiazek -- liczylaby sie podwojnie).
INDIVIDUAL_BOOK_PREFIXES = ["B365", "PS"] + POLISH_BOOK_PREFIXES


def add_derived_average_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Dolicza kolumny pochodne (nadpisywane przy kazdym zapisie, zawsze swieze):
      PolishAvg_winner/loser -- srednia z dostepnych polskich bukmacherow,
      AllAvg_winner/loser    -- srednia ze wszystkich pojedynczych bukmacherow
                                (B365, Pinnacle + polscy); 'EnglishAvg' (dawniej
                                'Avg') pozostaje srednia rynku zagranicznego.
    Srednie zaokraglane do 2 miejsc (bez tego artefakty floatow w CSV, np.
    1.6320000000000001). Migruje tez stare nazwy kolumn (Avg_* -> EnglishAvg_*),
    usuwa porzucone Max_* i normalizuje dtype match_num (dopisywanie wierszy
    upcastowalo int -> float i CSV mial '378.0').
    Srednia liczona z kolumn obecnych w ramce; NaN gdy zaden kurs nie istnieje."""
    df = df.rename(columns=LEGACY_COLUMN_RENAMES)
    df = df.drop(columns=[c for c in LEGACY_DROP_COLUMNS if c in df.columns])
    # Kolumny calkowitoliczbowe jako nullable Int64 (dopisywanie wierszy przez
    # pipeline BetExplorera wprowadza NaN i upcastuje int -> float: '378.0', '2.0').
    for int_col in ("match_num", "rank_agreement"):
        if int_col in df.columns:
            df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")
    # tourney_date w formacie ISO (YYYY-MM-DD); migruje tez stary zapis 20241230.
    if "tourney_date" in df.columns:
        raw_dates = df["tourney_date"].astype(str).str.replace(r"\.0$", "", regex=True)
        parsed = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")
        parsed = parsed.fillna(pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce"))
        df["tourney_date"] = parsed.dt.strftime("%Y-%m-%d")
    for side in ("winner", "loser"):
        polish_cols = [f"{p}_{side}" for p in POLISH_BOOK_PREFIXES
                       if f"{p}_{side}" in df.columns]
        df[f"PolishAvg_{side}"] = (df[polish_cols].mean(axis=1).round(2)
                                   if polish_cols else np.nan)
        all_cols = [f"{p}_{side}" for p in INDIVIDUAL_BOOK_PREFIXES
                    if f"{p}_{side}" in df.columns]
        df[f"AllAvg_{side}"] = (df[all_cols].mean(axis=1).round(2)
                                if all_cols else np.nan)
    return df

# Okno: data meczu tennis-data wzgledem tourney_date (startu turnieju) Sackmanna.
# -3 dni: turnieje startujace pod koniec grudnia maja czasem mecze przed oficjalna
# data; +27 dni: Grand Slam (15 dni) z zapasem.
DATE_WINDOW_DAYS = (-3, 27)

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (TenisPredictionModel research)"}


def matches_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


def odds_file(year: int) -> Path:
    return ODDS_DIR / f"{TOUR}_odds_{year}.csv"


def download_tennisdata_year(year: int, retries: int = 3) -> pd.DataFrame:
    """Pobiera roczny plik ATP z tennis-data.co.uk (xlsx, dla starych lat xls).

    Retry z backoffem -- przy kilku latach z rzedu serwer potrafi przejsciowo
    zerwac polaczenie.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        for ext in ("xlsx", "xls"):
            url = f"http://www.tennis-data.co.uk/{year}/{year}.{ext}"
            try:
                req = urllib.request.Request(url, headers=HTTP_HEADERS)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = resp.read()
                return pd.read_excel(io.BytesIO(payload))
            except Exception as exc:  # 404 dla zlego rozszerzenia, sieciowka itd.
                last_error = exc
        if attempt < retries - 1:
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Nie udalo sie pobrac kursow dla {year}: {last_error}")


def _normalize(text: str) -> str:
    """ASCII, lowercase, separatory (myslnik/apostrof/kropka) -> spacja."""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    for ch in "-'.":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _surname_variants(surname: str) -> set[str]:
    """Nazwisko ze spacjami i wariant 'sklejony' ('o connell' -> 'oconnell')."""
    variants = {surname}
    if " " in surname:
        variants.add(surname.replace(" ", ""))
    return variants


def tennisdata_player_keys(name: str) -> tuple[set[tuple[str, str]], set[str]]:
    """'Auger-Aliassime F.' -> ({(nazwisko, inicjal), ...}, {nazwisko, ...}).

    Format tennis-data: 'Nazwisko I.' lub 'Nazwisko I.I.' -- inicjaly to koncowe
    tokeny zawierajace kropke. Zwraca klucze (nazwisko, inicjal) oraz osobno
    same nazwiska (do fallbacku rankingowego).
    """
    raw_tokens = str(name).split()
    surname_tokens, initial = [], ""
    for token in raw_tokens:
        if "." in token:
            if not initial:
                initial = _normalize(token)[:1]
        else:
            surname_tokens.append(token)
    surname = _normalize(" ".join(surname_tokens))
    if not surname or not initial:
        return set(), set()
    surnames = _surname_variants(surname)
    return {(s, initial) for s in surnames}, surnames


def sackmann_player_keys(name: str) -> tuple[set[tuple[str, str]], set[str]]:
    """'Felix Auger Aliassime' -> ({(nazwisko, inicjal), ...}, {nazwisko, ...}).

    Tennis-data bywa niekonsekwentne, wiec generujemy szeroki zestaw kluczy:
      - wszystkie ciagle podciagi tokenow nazwiska (tokens[1:]) -- lapie
        "Auger-Aliassime F.", "Mpetshi G." i "Mpetshi Perricard G.";
      - porzadek odwrocony (nazwisko-pierwsze, inicjal z OSTATNIEGO tokenu) --
        lapie "Bu Y." dla Sackmannowego "Bu Yunchaokete";
      - warianty bez spacji ("Oconnell" vs "O Connell").
    Falszywym dopasowaniom zapobiega wymog zgodnosci obu graczy naraz + okno
    czasowe + tiebreak rankingowy.
    """
    tokens = _normalize(name).split()
    if len(tokens) < 2:
        return set(), set()

    keys: set[tuple[str, str]] = set()
    surnames: set[str] = set()

    initial = tokens[0][:1]
    n = len(tokens)
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            for s in _surname_variants(" ".join(tokens[i:j])):
                keys.add((s, initial))
                surnames.add(s)

    # Porzadek odwrocony: nazwisko jako prefiks, imie na koncu.
    reversed_initial = tokens[-1][:1]
    for i in range(1, n):
        for s in _surname_variants(" ".join(tokens[:i])):
            keys.add((s, reversed_initial))
            surnames.add(s)

    return keys, surnames


def build_tennisdata_index(td: pd.DataFrame) -> tuple[dict, dict]:
    """Dwa indeksy wierszy tennis-data:
    - scisly:   ((nazwisko_w, inicjal_w), (nazwisko_l, inicjal_l)) -> [idx, ...]
    - fallback: (nazwisko_w, nazwisko_l)                            -> [idx, ...]
    """
    strict_index: dict[tuple, list[int]] = defaultdict(list)
    surname_index: dict[tuple, list[int]] = defaultdict(list)
    for row_idx, (w_name, l_name) in enumerate(zip(td["Winner"], td["Loser"])):
        w_keys, w_surnames = tennisdata_player_keys(w_name)
        l_keys, l_surnames = tennisdata_player_keys(l_name)
        if not w_keys or not l_keys:
            continue
        for wk in w_keys:
            for lk in l_keys:
                strict_index[(wk, lk)].append(row_idx)
        for ws in w_surnames:
            for ls in l_surnames:
                surname_index[(ws, ls)].append(row_idx)
    return strict_index, surname_index


def _rank_agreement(td_row, winner_rank, loser_rank) -> int:
    """Ile z dwoch rankingow (W/L) zgadza sie miedzy zrodlami (0-2)."""
    score = 0
    for td_col, sack_val in (("WRank", winner_rank), ("LRank", loser_rank)):
        td_val = pd.to_numeric(pd.Series([td_row.get(td_col)]), errors="coerce").iloc[0]
        if pd.notna(td_val) and pd.notna(sack_val) and int(td_val) == int(sack_val):
            score += 1
    return score


def match_year(year: int, verbose: bool = True) -> pd.DataFrame:
    """Laczy mecze Sackmanna z kursami tennis-data dla jednego sezonu."""
    sack = pd.read_csv(matches_file(year))
    sack["tourney_date"] = pd.to_datetime(sack["tourney_date"], format="%Y%m%d")

    td = download_tennisdata_year(year)
    td["Date"] = pd.to_datetime(td["Date"])
    available_books = {out: cols for out, cols in ODDS_COLUMNS.items()
                       if cols[0] in td.columns and cols[1] in td.columns}
    if verbose:
        print(f"  tennis-data: {len(td)} meczow | kursy: {sorted(available_books)}")

    strict_index, surname_index = build_tennisdata_index(td)
    used_td_rows: set[int] = set()
    out_rows = []

    lo, hi = DATE_WINDOW_DAYS

    def best_candidate(candidate_ids, sack_row, min_agreement=0):
        best = None  # (rank_agreement_neg, date_diff, td_idx)
        for td_idx in set(candidate_ids):
            if td_idx in used_td_rows:
                continue
            td_row = td.iloc[td_idx]
            day_diff = (td_row["Date"] - sack_row.tourney_date).days
            if not (lo <= day_diff <= hi):
                continue
            agreement = _rank_agreement(td_row, sack_row.winner_rank, sack_row.loser_rank)
            if agreement < min_agreement:
                continue
            key = (-agreement, abs(day_diff), td_idx)
            if best is None or key < best:
                best = key
        return best

    for sack_row in sack.itertuples(index=False):
        w_keys, w_surnames = sackmann_player_keys(sack_row.winner_name)
        l_keys, l_surnames = sackmann_player_keys(sack_row.loser_name)

        candidates: list[int] = []
        for wk in w_keys:
            for lk in l_keys:
                candidates.extend(strict_index.get((wk, lk), []))
        best = best_candidate(candidates, sack_row)

        if best is None:
            # Fallback na rozjazd inicjalow (np. "Barrios M." = Marcelo Tomas
            # Barrios Vera): same nazwiska pary, ale wymagamy pelnej zgodnosci
            # OBU rankingow -- to praktycznie wyklucza falszywe dopasowanie.
            fallback: list[int] = []
            for ws in w_surnames:
                for ls in l_surnames:
                    fallback.extend(surname_index.get((ws, ls), []))
            best = best_candidate(fallback, sack_row, min_agreement=2)
        if best is None:
            continue

        td_idx = best[2]
        used_td_rows.add(td_idx)
        td_row = td.iloc[td_idx]

        record = {
            "match_key": f"{sack_row.tourney_id}_{sack_row.match_num}",
            "tourney_id": sack_row.tourney_id,
            "match_num": sack_row.match_num,
            "tourney_date": sack_row.tourney_date.strftime("%Y-%m-%d"),
            "winner_name": sack_row.winner_name,
            "loser_name": sack_row.loser_name,
            "td_date": td_row["Date"].strftime("%Y-%m-%d"),
            "td_tournament": td_row.get("Tournament", ""),
            "rank_agreement": -best[0],
        }
        for out_name, (w_col, l_col) in available_books.items():
            record[f"{out_name}_winner"] = td_row[w_col]
            record[f"{out_name}_loser"] = td_row[l_col]
        out_rows.append(record)

    result = add_derived_average_columns(pd.DataFrame(out_rows))
    if verbose:
        total = len(sack)
        n = len(result)
        print(f"  dopasowano kursy: {n}/{total} meczow ({n / total * 100:.1f}%)")
        unmatched = sack[~(sack["tourney_id"].astype(str) + "_" + sack["match_num"].astype(str))
                         .isin(result["match_key"] if n else [])]
        if len(unmatched):
            top = unmatched.groupby("tourney_name").size().sort_values(ascending=False).head(5)
            print("  najwiecej bez kursow (turniej: liczba meczow):")
            for tname, cnt in top.items():
                print(f"    {tname}: {cnt}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Pobiera kursy tennis-data.co.uk i "
                                             "dopasowuje do atp_matches_{rok}.csv")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help=f"Lata (default: {DEFAULT_YEARS}).")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Pauza miedzy latami w sekundach (grzecznosciowa, default 1.0).")
    ap.add_argument("--force-refetch", action="store_true",
                    help="Przelicz rok nawet jesli data/odds/atp_odds_{rok}.csv istnieje.")
    ap.add_argument("--clean", action="store_true",
                    help="USUN istniejace pliki atp_odds_{rok}.csv dla podanych lat "
                         "przed pobraniem (pelny reset).")
    ap.add_argument("--recompute-averages", action="store_true",
                    help="Bez pobierania: przelicz kolumny pochodne (PolishAvg_*, "
                         "AllAvg_*) w istniejacych plikach atp_odds_{rok}.csv. "
                         "Uzyteczne po doladowaniu polskich kursow z BetExplorer.")
    args = ap.parse_args()

    years = args.years or DEFAULT_YEARS
    ODDS_DIR.mkdir(parents=True, exist_ok=True)

    if args.recompute_averages:
        for year in years:
            fp = odds_file(year)
            if not fp.exists():
                print(f"[{year}] brak {fp.name} -- pomijam")
                continue
            df = add_derived_average_columns(pd.read_csv(fp))
            # Sprzatanie: jeden mecz BetExplorer nie moze byc przypisany do
            # dwoch wierszy (np. dwa rubbery tej samej pary w Davis Cup --
            # zostawiamy pierwsze przypisanie, drugie czyscimy).
            if "be_match_id" in df.columns:
                dup = df["be_match_id"].notna() & df["be_match_id"].duplicated(keep="first")
                if dup.any():
                    print(f"[{year}] czyszcze {int(dup.sum())} zduplikowanych be_match_id")
                    df.loc[dup, "be_match_id"] = pd.NA
            df.to_csv(fp, index=False)
            n_pol = int(df["PolishAvg_winner"].notna().sum())
            n_all = int(df["AllAvg_winner"].notna().sum())
            print(f"[{year}] przeliczono: PolishAvg dla {n_pol}, AllAvg dla {n_all} "
                  f"z {len(df)} wierszy")
        return

    if args.clean:
        for year in years:
            fp = odds_file(year)
            if fp.exists():
                print(f"[--clean] usuwam {fp}")
                fp.unlink()

    print("=" * 78)
    print(" KURSY BUKMACHERSKIE (tennis-data.co.uk) -> data/odds/")
    print(f"  Lata : {years}")
    print("=" * 78)

    for i, year in enumerate(years):
        print(f"\n[Rok {year}]")
        out_path = odds_file(year)
        if out_path.exists() and not args.force_refetch:
            existing = pd.read_csv(out_path)
            print(f"  pomijam -- {out_path.name} juz istnieje ({len(existing)} meczow); "
                  f"uzyj --force-refetch aby przeliczyc")
            continue
        if not matches_file(year).exists():
            print(f"  UWAGA: brak {matches_file(year).name} -- pomijam rok.")
            continue
        result = match_year(year)
        result.to_csv(out_path, index=False)
        print(f"  zapisano: {out_path}")
        if i < len(years) - 1 and args.sleep > 0:
            time.sleep(args.sleep)

    print()
    print("=" * 78)
    print(" DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
