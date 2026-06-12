"""
Polskie kursy bukmacherskie z BetExplorer (ukryte API, bez Playwright)
======================================================================

Dolepia kursy polskich bukmacherów do data/odds/atp_odds_{rok}.csv (po match_key),
obok międzynarodowych kursów z tennis-data (src/fetch_odds.py). Dostępność
historyczna (zbadana empirycznie): 2020 — Betclic.pl/eFortuna.pl/LV BET/STS.pl;
od 2022 dochodzą BETFAN i Superbet.pl; od 2025 Fuksiarz.pl.

Jak to działa (czyste HTTP GET, BetExplorer nie wymaga Playwright/Cloudflare):
  1) ENUMERACJA ID MECZÓW: dzienne strony wyników
       https://www.betexplorer.com/tennis/results/?year=Y&month=M&day=D
     zawierają linki /tennis/atp-singles/{turniej}/{gracze}/{ID8}/. Dni do
     pobrania znamy z td_date (realna data meczu z tennis-data, kolumna w
     atp_odds_{rok}.csv) ±1 dzień; dla meczów bez td_date skanujemy okno
     [tourney_date .. tourney_date+14]. Strony dzienne są cache'owane w runie.
  2) KURSY: endpoint JSON (ten sam, którego używa frontend strony meczu)
       https://www.betexplorer.com/match-odds-old/{ID}/1/ha/0/en/
     zwraca {"odds": "<html tabeli>"} z wierszami bukmacherów widocznymi dla
     polskiego IP (geo-detekcja). Parsujemy nazwę bukmachera + data-odd
     (dwa kursy: gracz lewy / prawy), orientację mapujemy na zwycięzcę/
     przegranego po nazwiskach.
  3) ZAPIS: kolumny {Bukmacher}_winner/{Bukmacher}_loser + be_match_id
     dolepiane do atp_odds_{rok}.csv po match_key. Mecze, których nie ma
     w pliku (np. ATP Cup / United Cup — BetExplorer je MA, tennis-data nie),
     dopisywane są jako nowe wiersze (kolumny międzynarodowe puste).
     Flush co FLUSH_EVERY meczów -> Ctrl+C nie traci postępu; pre-fetch dedup
     pomija mecze z już pobranym be_match_id (chyba że --force-refetch).

Dopasowanie nazwisk reużywa maszynerii z src/fetch_odds.py (BetExplorer pokazuje
"Medvedev D." — ten sam format co tennis-data). Przy niejednoznaczności para
jest pomijana z ostrzeżeniem.

Uruchomienie:
  python src/fetch_odds_betexplorer.py --years 2025
  python src/fetch_odds_betexplorer.py --years 2020 2021 2022 2023 2024 --sleep 1.5
  python src/fetch_odds_betexplorer.py --years 2025 --limit 30          # pilot
  python src/fetch_odds_betexplorer.py --years 2025 --force-refetch

Tempo: ~1.5 s/mecz przy --sleep 1.0 (lekki GET zamiast 4-5 s Playwright)
-> pełny sezon ~2600 meczów ≈ 60-70 min + ~10 min enumeracji dni.
Zwiększ --sleep, jeśli serwer zacznie odrzucać żądania (429/503).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_odds import (  # noqa: E402
    ODDS_DIR,
    add_derived_average_columns,
    matches_file,
    odds_file,
    sackmann_player_keys,
    tennisdata_player_keys,
)

BASE_URL = "https://www.betexplorer.com"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Referer": "https://www.betexplorer.com/",
    "X-Requested-With": "XMLHttpRequest",
}

# Nazwa bukmachera na BetExplorer -> prefiks kolumny (zgodny z oryginalnym
# pipeline'em polskich kursów). Nieznane nazwy dostają prefiks zsanityzowany.
BOOKMAKER_COLUMNS = {
    "Betclic.pl": "Betclic", "BETFAN": "BETFAN", "eFortuna.pl": "Fortuna",
    "Fuksiarz.pl": "Fuksiarz", "LV BET": "LVBET", "STS.pl": "STS",
    "Superbet.pl": "Superbet",
}

DAY_WINDOW_NO_TDDATE = 14   # skan dni od tourney_date dla meczow bez td_date
FLUSH_EVERY = 10            # zapis czastkowy co N pobranych meczow

MATCH_LINK_RE = re.compile(
    r'<a[^>]*href="(/tennis/atp-singles/[a-z0-9-]+/[a-z0-9-]+/([A-Za-z0-9]{8})/)"[^>]*>(.*?)</a>',
    re.S,
)
# Nazwy graczy na stronie dziennej siedza w dwoch spanach (bez separatora " - "):
#   <span class="...teamLine--home"><strong>Guinard M.</strong></span>
#   <span class="...teamLine--away">Daniel T.</span>
TEAM_HOME_RE = re.compile(r'teamLine--home[^>]*>(.*?)</span>', re.S)
TEAM_AWAY_RE = re.compile(r'teamLine--away[^>]*>(.*?)</span>', re.S)
ODDS_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
BOOKIE_NAME_RE = re.compile(r">([^<>]+)</a>")
DATA_ODD_RE = re.compile(r'data-odd="([\d.]+)"')


def http_get(url: str, retries: int = 3, sleep_s: float = 2.0) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HTTP_HEADERS)
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"GET {url} nieudany po {retries} probach: {last_error}")


class DayIndex:
    """Lazy cache dziennych stron wynikow: dzien -> lista meczow ATP singles.

    Kazdy mecz: (match_id, klucze_gracza_lewego, klucze_gracza_prawego),
    gdzie klucze to zbior (nazwisko, inicjal) jak w fetch_odds.
    """

    def __init__(self, sleep_between: float):
        self.sleep_between = sleep_between
        self._cache: dict[date, list[tuple[str, set, set]]] = {}
        self.pages_fetched = 0

    def matches_for_day(self, day: date) -> list[tuple[str, set, set]]:
        if day in self._cache:
            return self._cache[day]
        url = f"{BASE_URL}/tennis/results/?year={day.year}&month={day.month}&day={day.day}"
        body = http_get(url)
        self.pages_fetched += 1
        entries: dict[str, tuple[str, set, set]] = {}
        for _href, match_id, inner in MATCH_LINK_RE.findall(body):
            home_m = TEAM_HOME_RE.search(inner)
            away_m = TEAM_AWAY_RE.search(inner)
            if not home_m or not away_m:
                continue  # link wynikowy/duplikat bez nazw graczy
            left = " ".join(re.sub(r"<[^>]+>", " ", home_m.group(1)).split())
            right = " ".join(re.sub(r"<[^>]+>", " ", away_m.group(1)).split())
            left_keys, _ = tennisdata_player_keys(left)
            right_keys, _ = tennisdata_player_keys(right)
            if left_keys and right_keys:
                entries[match_id] = (match_id, left_keys, right_keys)
        self._cache[day] = list(entries.values())
        time.sleep(self.sleep_between)
        return self._cache[day]


def candidate_days(sack_row, td_date_by_key: dict[str, str], match_key: str) -> list[date]:
    td_date = td_date_by_key.get(match_key)
    if td_date:
        base = pd.Timestamp(td_date).date()
        return [base, base - timedelta(days=1), base + timedelta(days=1)]
    start = pd.Timestamp(sack_row.tourney_date).date()
    return [start + timedelta(days=i) for i in range(DAY_WINDOW_NO_TDDATE + 1)]


def find_betexplorer_match(sack_row, days: list[date], day_index: DayIndex,
                           used_ids: set[str]):
    """Zwraca (match_id, winner_is_left) albo None. Niejednoznacznosc -> None.

    used_ids: ID meczow BetExplorer juz przypisanych innym wierszom -- jeden
    mecz BetExplorer odpowiada najwyzej jednemu meczowi Sackmanna (inaczej
    np. dwa rubbery tej samej pary w Davis Cup dostalyby ten sam ID)."""
    w_keys, _ = sackmann_player_keys(sack_row.winner_name)
    l_keys, _ = sackmann_player_keys(sack_row.loser_name)
    found: dict[str, bool] = {}
    for day in days:
        for match_id, left_keys, right_keys in day_index.matches_for_day(day):
            if match_id in used_ids:
                continue
            winner_left = bool(w_keys & left_keys) and bool(l_keys & right_keys)
            winner_right = bool(w_keys & right_keys) and bool(l_keys & left_keys)
            if winner_left and winner_right:
                return None  # np. bracia o tym samym inicjale -- nie zgadujemy
            if winner_left or winner_right:
                found[match_id] = winner_left
    if len(found) != 1:
        return None  # brak albo wiele roznych meczow (powtorka pary w oknie)
    return next(iter(found.items()))


def fetch_polish_odds(match_id: str) -> dict[str, tuple[float, float]]:
    """Kursy z /match-odds-old: {nazwa_bukmachera: (kurs_lewego, kurs_prawego)}."""
    url = f"{BASE_URL}/match-odds-old/{match_id}/1/ha/0/en/"
    payload = json.loads(http_get(url))
    odds_html = payload.get("odds", "")
    result: dict[str, tuple[float, float]] = {}
    for row in ODDS_ROW_RE.findall(odds_html):
        name_match = BOOKIE_NAME_RE.search(row)
        odds = DATA_ODD_RE.findall(row)
        if name_match and len(odds) >= 2:
            result[name_match.group(1).strip()] = (float(odds[0]), float(odds[1]))
    return result


def bookmaker_prefix(name: str) -> str:
    if name in BOOKMAKER_COLUMNS:
        return BOOKMAKER_COLUMNS[name]
    return re.sub(r"[^A-Za-z0-9]+", "", name) or "Bookie"


def flush(odds_df: pd.DataFrame, path: Path) -> None:
    # Kolumny pochodne (PolishAvg_*, AllAvg_*) przeliczane przy kazdym zapisie,
    # zeby srednie obejmowaly swiezo dolepione polskie kursy.
    odds_df = add_derived_average_columns(odds_df.copy())
    tmp = path.with_suffix(path.suffix + ".tmp")
    odds_df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def process_year(year: int, sleep_between: float, force_refetch: bool,
                 limit: int | None) -> None:
    sack = pd.read_csv(matches_file(year))
    sack["tourney_date"] = pd.to_datetime(sack["tourney_date"], format="%Y%m%d")
    sack["match_key"] = sack["tourney_id"].astype(str) + "_" + sack["match_num"].astype(str)

    out_path = odds_file(year)
    if out_path.exists():
        odds_df = pd.read_csv(out_path)
        # Kolumny identyfikacyjne jako stringi -- read_csv wczytuje tourney_date
        # jako int, a dopisywane wiersze maja stringi (TypeError przy coercion).
        for col in ("tourney_id", "tourney_date", "match_key"):
            if col in odds_df.columns:
                odds_df[col] = odds_df[col].astype(str)
    else:
        print(f"  UWAGA: brak {out_path.name} (kursy miedzynarodowe) -- tworze nowy "
              f"plik z samymi polskimi kursami. Zalecane najpierw: python src/fetch_odds.py")
        odds_df = pd.DataFrame(columns=["match_key", "tourney_id", "match_num",
                                        "tourney_date", "winner_name", "loser_name"])
    odds_df = odds_df.set_index("match_key", drop=False)
    td_date_by_key = (odds_df["td_date"].dropna().to_dict()
                      if "td_date" in odds_df.columns else {})

    if "be_match_id" not in odds_df.columns:
        # Jawny dtype=object -- przypisanie pd.NA tworzy kolumne float64,
        # ktora odrzuca stringowe ID (TypeError przy coercion).
        odds_df["be_match_id"] = pd.Series(pd.NA, index=odds_df.index, dtype="object")
    else:
        odds_df["be_match_id"] = odds_df["be_match_id"].astype("object")

    day_index = DayIndex(sleep_between)
    done = fetched = not_found = 0
    skipped_existing = 0
    # ID juz przypisane (z pliku) -- przy force-refetch budujemy od zera,
    # zeby wiersz mogl ponownie dopasowac swoj wlasny mecz.
    used_be_ids: set[str] = (set() if force_refetch
                             else set(odds_df["be_match_id"].dropna().astype(str)))

    for sack_row in sack.itertuples(index=False):
        match_key = sack_row.match_key
        already = (match_key in odds_df.index
                   and pd.notna(odds_df.at[match_key, "be_match_id"]))
        if already and not force_refetch:
            skipped_existing += 1
            continue

        days = candidate_days(sack_row, td_date_by_key, match_key)
        located = find_betexplorer_match(sack_row, days, day_index, used_be_ids)
        if located is None:
            not_found += 1
            continue
        match_id, winner_is_left = located
        used_be_ids.add(match_id)

        polish = fetch_polish_odds(match_id)
        time.sleep(sleep_between)
        fetched += 1

        if match_key not in odds_df.index:
            odds_df.loc[match_key, ["match_key", "tourney_id", "match_num",
                                    "tourney_date", "winner_name", "loser_name"]] = [
                match_key, sack_row.tourney_id, sack_row.match_num,
                sack_row.tourney_date.strftime("%Y%m%d"),
                sack_row.winner_name, sack_row.loser_name,
            ]
        odds_df.at[match_key, "be_match_id"] = match_id
        for bookie_name, (left_odd, right_odd) in polish.items():
            prefix = bookmaker_prefix(bookie_name)
            w_odd, l_odd = (left_odd, right_odd) if winner_is_left else (right_odd, left_odd)
            for col in (f"{prefix}_winner", f"{prefix}_loser"):
                if col not in odds_df.columns:
                    odds_df[col] = np.nan  # kursy = float
            odds_df.at[match_key, f"{prefix}_winner"] = w_odd
            odds_df.at[match_key, f"{prefix}_loser"] = l_odd

        done += 1
        if done % FLUSH_EVERY == 0:
            flush(odds_df.reset_index(drop=True), out_path)
            print(f"  ...{done} meczow z BetExplorer (flush), stron dziennych: "
                  f"{day_index.pages_fetched}", flush=True)
        if limit is not None and done >= limit:
            print(f"  --limit {limit} osiagniety, przerywam rok.")
            break

    flush(odds_df.reset_index(drop=True), out_path)
    polish_cols = [c for c in odds_df.columns if c.endswith("_winner")
                   and c.split("_")[0] in set(BOOKMAKER_COLUMNS.values())]
    with_polish = odds_df[polish_cols].notna().any(axis=1).sum() if polish_cols else 0
    print(f"  ROK {year}: pobrano teraz {fetched}, pominieto (juz mialy) {skipped_existing}, "
          f"nie znaleziono na BetExplorer {not_found}")
    print(f"  lacznie wierszy z polskimi kursami w {out_path.name}: {with_polish}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dolepia polskie kursy z BetExplorer "
                                             "do data/odds/atp_odds_{rok}.csv")
    ap.add_argument("--years", type=int, nargs="+", default=[2025],
                    help="Lata (default: 2025).")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Pauza miedzy requestami (default 1.0 s). Zwieksz przy 429/503.")
    ap.add_argument("--force-refetch", action="store_true",
                    help="Re-scrape takze meczow, ktore juz maja be_match_id.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Pobierz najwyzej N meczow na rok (pilot/debug).")
    args = ap.parse_args()

    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(" POLSKIE KURSY (BetExplorer, ukryte API match-odds-old)")
    print(f"  Lata : {args.years} | sleep {args.sleep}s | "
          f"{'force-refetch' if args.force_refetch else 'dedup wlaczony'}")
    print("=" * 78)

    for year in args.years:
        print(f"\n[Rok {year}]")
        if not matches_file(year).exists():
            print(f"  UWAGA: brak {matches_file(year).name} -- pomijam.")
            continue
        process_year(year, args.sleep, args.force_refetch, args.limit)

    print()
    print("=" * 78)
    print(" DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
