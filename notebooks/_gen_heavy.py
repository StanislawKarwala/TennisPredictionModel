"""Generuje i wykonuje OD ZERA ciezkie notebooki (jeden na wywolanie).

Uzycie: python _gen_heavy.py <nazwa>
gdzie nazwa in {elo, multiseason, walkforward, salvage, validate_variants, validate_features}.

Wzorzec: markdown z metoda + komorka uruchamiajaca caly eksperyment od zera
(import modulu i wywolanie main(), ktore drukuje realne wyniki) + wnioski.
"""
import sys
from _nbtools import make_and_run

SETUP = """import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))"""

SPECS = {
"elo": ("TPM_Experiment_Elo.ipynb", 3000, [
("md", """# Eksperyment: Surface-adjusted Elo (Sprint 5)

## Cel
Dodac do baseline rating **Elo** (jak w szachach) -- przewidujacy, aktualizowany wynikami meczow,
w przeciwienstwie do rankingu ATP (suma punktow do rozstawiania). Dwa ratingi: ogolny + per
nawierzchnia. Cztery cechy: `elo_diff`, `surface_elo_diff`, `elo_win_prob`, `surface_elo_win_prob`.

## Metoda
Elo liczony SAMODZIELNIE z danych Sackmanna -- z natury sekwencyjny (expanding window), wiec
leakage-safe. K-factor dynamiczny (FiveThirtyEight). **Walidacja walk-forward** przez kilka sezonow
+ test parowany McNemar (lekcja: pojedynczy test klamie)."""),
("code", SETUP),
("md", "## Uruchomienie pelnej walidacji walk-forward (od zera)\nKazdy sezon: baseline + model z Elo na tych samych meczach, parowanie per-mecz."),
("code", "import main_48_cech_elo as m\nm.main()"),
("md", """## Wnioski
Cechy Elo **dominuja waznosc** (elo_diff to top-2 cecha) -- model mocno ich uzywa. ALE pooled delta
jest mala (rzedu +0.5 p.p.) i **nieistotna** (McNemar p >> 0.05). Powod: Elo jest silnym, ale
REDUNDANTNYM sygnalem -- baseline ma juz ranking ATP, ktory mierzy to samo. Literaturowe "Elo ~70%"
dotyczy Elo jako GLOWNEGO sygnalu, nie dodatku do modelu z rankingiem. Elo zaplaciloby dopiero przy
duzo dluzszej historii do rozgrzewki ratingow."""),
]),

"multiseason": ("TPM_Experiment_MultiSeason.ipynb", 7200, [
("md", """# Eksperyment: Wielo-sezonowy trening + RF vs HGB vs XGBoost (Sprint 6)

## Cel
Bazowa architektura trenuje TYLKO na roku docelowym (~3500 probek), wiec gradient boosting nie ma
jak rozwinac przewagi. Tu trenujemy na WIELU sezonach (domyslnie 2010-2023, ~72 tys. probek),
walidujemy na 2024, testujemy na calym sezonie 2025 -- i porownujemy Random Forest vs
HistGradientBoosting vs XGBoost. Wlasciwy test hipotezy "wiecej danych => boosting wygrywa"."""),
("code", SETUP),
("md", "## Uruchomienie (od zera): wczytanie wielu sezonow, cechy, strojenie 3 modeli\nUWAGA: dlugi bieg (~60-90 min) -- liczenie cech dla ~45 tys. meczow + trening na ~72 tys. probek."),
("code", "import main_48_cech_multiseason as m\nm.main()"),
("md", """## Wnioski
Boosting **nie pobil** Random Forest na accuracy nawet na ~72 tys. probek (XGBoost remis, HGB
minimalnie gorzej) -- wszystkie ~65%. Powtorzono tez na ~130 tys. probek (od 2000) -- ten sam
wynik. 3 algorytmy x zakres danych od 3.5k do 130k probek -> wszystko ~65%. **Sciana jest w
cechach/problemie, nie w algorytmie ani ilosci danych.** Jedyna roznica: XGBoost ma minimalnie
lepsza kalibracje (Brier/log-loss), co rosnie z iloscia danych -- istotne tylko gdyby celem byl
betting, nie accuracy."""),
]),

"walkforward": ("TPM_Experiment_WalkForward.ipynb", 5400, [
("md", """# Walidacja: Walk-forward modelu wzbogaconego (Sprint 4)

## Cel
Najwazniejszy krok metodologiczny. Cechy ze Sprintu 3 (surface_speed + fatigue) na pojedynczym
sezonie dawaly +2 p.p. Tu sprawdzamy je UCZCIWIE: trening na starszych sezonach, test na kolejnych,
przez kilka lat z rzedu, z testem istotnosci McNemar (parowanie per-mecz baseline vs wzbogacony)."""),
("code", SETUP),
("md", "## Uruchomienie walk-forward (od zera, kilka sezonow)"),
("code", "import main_48_cech_walkforward as m\nm.main()"),
("md", """## Wnioski
Pozorny zysk +2 p.p. **NIE generalizuje** -- pooled delta blisko zera, McNemar p ~ 0.9 (zero
istotnosci), delta dodatnia tylko w czesci lat. To kluczowa lekcja: na malych danych pojedynczy
test set potrafi pokazac przekonujacy, ale nieprawdziwy zysk. Dopiero wieloletnia walidacja mowi
prawde."""),
]),

"salvage": ("TPM_Experiment_Salvage.ipynb", 5400, [
("md", """# Analiza salvage: czy WEZSZY zestaw cech generalizuje? (Sprint 4)

## Cel
Skoro pelny zestaw 9 cech (surface+fatigue) to szum, moze wezszy podzbior (sama interakcja
serve x speed) generalizuje lepiej? Testujemy 4 warianty (full / speed3 / narrow2 / single1) na
identycznych meczach, walk-forward, McNemar."""),
("code", SETUP),
("md", "## Uruchomienie (od zera)"),
("code", "import main_48_cech_salvage as m\nm.main()"),
("md", """## Wnioski
Zaden wariant -- nawet najwezszy -- nie daje istotnego zysku (wszystkie McNemar p > 0.3). Wezsze
zestawy sa minimalnie lepsze od pelnego (potwierdza "za duzo cech rozciencza"), ale wciaz w
granicach szumu. Ostateczne potwierdzenie: cechy surface/fatigue nie daja robust zysku."""),
]),

"validate_variants": ("TPM_Experiment_ValidateVariants.ipynb", 6000, [
("md", """# Walidacja: warianty slice-aware (bestof5 / qfserve / sliceaware) (Sprint 4+)

## Cel
Warianty slice-aware byly oceniane tylko na pojedynczym tescie (gdzie bestof5 dawal +2.37, qfserve
+2.20). Tu sprawdzamy je UCZCIWIE -- walk-forward przez kilka sezonow + McNemar. Baseline liczony
raz na rok (cache), warianty go reuzywaja."""),
("code", SETUP),
("md", "## Uruchomienie (od zera) -- baseline + 3 warianty na kazdy sezon"),
("code", "import main_48_cech_validate_variants as m\nm.main()"),
("md", """## Wnioski
ZADEN wariant slice-aware nie pobil baseline w sposob istotny (wszystkie McNemar p > 0.18).
Dawne "spektakularne" wyniki (bestof5 +2.37, qfserve +2.20) na walk-forward spadly do szumu --
qfserve nawet na minus. **Na wiekszej ilosci danych warianty bestof5 i qfserve okazaly sie slabsze
od glownego modelu `main_48_cech.py`.** To ostateczne potwierdzenie, ze pojedynczy test set klamie."""),
]),

"validate_features": ("TPM_Experiment_ValidateFeatures.ipynb", 5400, [
("md", """# Walidacja: zestawy cech na nowych danych (Sprint 4+)

## Cel
Spojna, swieza walidacja walk-forward wszystkich zestawow cech na nowych danych: surface_speed,
fatigue, enriched (surface+fatigue), Elo. Per sezon: baseline (cache) + kazdy zestaw na tych samych
meczach, parowanie + McNemar."""),
("code", SETUP),
("md", "## Uruchomienie (od zera)"),
("code", "import main_48_cech_validate_features as m\nm.main()"),
("md", """## Wnioski
ZADEN zestaw cech nie jest istotny statystycznie (wszystkie McNemar p > 0.25). Najlepsze sa
directionally dodatnie (+0.4-0.6 p.p.), ale w granicach szumu. To komplet dowodow, ze nic nie
przebija baseline -- sufit ~65% jest odporny na nowe cechy."""),
]),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SPECS:
        print("Uzycie: python _gen_heavy.py <" + "|".join(SPECS) + ">")
        sys.exit(1)
    name, timeout, cells = SPECS[sys.argv[1]]
    make_and_run(name, cells, timeout=timeout)


if __name__ == "__main__":
    main()
