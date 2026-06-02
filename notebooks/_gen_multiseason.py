"""Generuje i wykonuje OD ZERA notebook wielo-sezonowy (Sprint 6) w stylu
narracyjnym (jak TPM_Experiment_SliceAware_BestOf5_v1.ipynb) -- markdown opisuje
krok, kod go WYKONUJE i drukuje posrednie wyniki, zamiast wolac m.main().

Reuzywamy publiczne funkcje z tennis_model_multiseason.py (load_years,
add_static_features, tune_and_eval, run_baseline_quietly) -- NIE duplikujemy
logiki ani nie wolamy main(). Cechy sa IDENTYCZNE z baseline (przez namespace),
splity po roku PLIKU (season), seed 42 -- wszystko leakage-safe jak w module.

Trening od 2001 (~128 tys. probek symetryzowanych); sezon 2000 = rozgrzewka cech.

Uzycie: python _gen_multiseason.py
"""
from _nbtools import make_and_run

SETUP = """import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))"""

cells = [
("md", """# Eksperyment: Wielo-sezonowy trening + uczciwy test boostingu (Sprint 6)

## Cel
Dotychczasowa architektura trenowala **tylko na roku docelowym** (~3500 probek). Sprint 2 pokazal,
ze HistGradientBoosting nie bije Random Forest -- ale na tak malej probie boosting nie ma jak
rozwinac swojej przewagi. Tutaj zmieniamy architekture: trenujemy na **wielu sezonach**
(2001-2023, ~128 tys. probek po symetryzacji -- praktycznie cale ATP od 2000), walidujemy na 2024 i
testujemy na **calym sezonie 2025**. To wlasciwy test hipotezy *"wiecej danych => boosting wreszcie
oplacalny"*.

## Metoda (leakage-safe, ten sam matrix dla wszystkich modeli)
- **Cechy IDENTYCZNE z baseline** (40 cech) -- reuzywamy `add_dynamic_features` / `symmetrize_data` z
  `tennis_model.py` przez namespace. Jedyne zmienne to **ilosc danych treningowych** i **algorytm**.
- **Rozgrzewka cech: sezon 2000** -- liczy historie (forma, H2H, surface form) dla pierwszych meczow
  2001, ale **nie wchodzi** do zbioru treningowego.
- **Split po roku PLIKU** (`season`), nie po `tourney_date.dt.year`: plik sezonu 2025 zaczyna sie od
  United Cup z konca grudnia 2024, wiec sama data myli sezon.
- **Label encoding** fitowany TYLKO na treningu (`season < 2024`) -- zero wgladu w val/test.
- **CV chronologiczne** (`TimeSeriesSplit`) + tuning po `neg_log_loss`, ten sam `random_state=42`.
- Trzy modele -- **RF vs HistGradientBoosting vs XGBoost** -- na **dokladnie tej samej** macierzy
  cech, porownanie match accuracy oraz jakosci kalibracji (Brier / log-loss / ECE).

> UWAGA: to **pojedynczy** trening wielo-sezonowy (jeden train/val/test), a **nie** walk-forward.
> Baseline tego notebooka (~0.647 na 2647 meczach 2025) to **inne dane** niz single-season 0.6566 --
> nie porownujemy ich 1:1."""),

("code", SETUP),

("md", """## 1. Reuse baseline -- pobieramy funkcje feature-engineering
Najpierw ustawiamy zakres treningu (**od 2001**, rozgrzewka 2000) przez zmienne srodowiskowe -- modul
czyta je przy imporcie. Potem uruchamiamy `tennis_model.py` raz (z wyciszonym outputem) i wyciagamy z
jego namespace funkcje: budowanie cech dynamicznych, symetryzacja, symetryczna metryka match-level
oraz ocena kalibracji. Bierzemy tez liste **40 cech** i `cols_base` -- macierz jest taka sama jak w
baseline, tylko zbudowana na wielu sezonach."""),

("code", """import os
# Trening od 2001 (~128 tys. probek), sezon 2000 jako rozgrzewka cech -- pelny zakres danych od 2000.
# Ustawiamy PRZED importem modulu (czyta env przy imporcie).
os.environ["TENNIS_WARMUP_START"] = "2000"
os.environ["TENNIS_TRAIN_START"] = "2001"

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import tennis_model_multiseason as m

WARMUP_START = m.WARMUP_START
TRAIN_START  = m.TRAIN_START
VAL_YEAR     = m.VAL_YEAR
TEST_YEAR    = m.TEST_YEAR
RANDOM_STATE = m.RANDOM_STATE

print("Uruchamiam baseline raz (pobranie funkcji feature-engineering)...")
ns = m.run_baseline_quietly()                      # runpy tennis_model.py (cicho)
add_dynamic_features = ns["add_dynamic_features"]
symmetrize_data = ns["symmetrize_data"]
compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
evaluate_calibration_quality = ns["evaluate_calibration_quality"]
features = list(ns["features"])
cols_base = list(ns["cols_base"])
ROUND_ORDER = ns["ROUND_ORDER"]

print(f"\\nTrening {TRAIN_START}-{VAL_YEAR-1} | walidacja {VAL_YEAR} | test {TEST_YEAR}")
warmup_desc = f"{WARMUP_START}-{TRAIN_START-1}" if TRAIN_START > WARMUP_START else "BRAK"
print(f"Rozgrzewka cech: {warmup_desc}")
print(f"Liczba cech (identyczna z baseline): {len(features)}")
print(f"XGBoost dostepny: {m.HAS_XGB}")"""),

("md", """## 2. Wczytujemy wszystkie sezony 2000-2025 i dzielimy na rozgrzewke / span
`load_years` czyta pliki `atp_matches_<rok>.csv`, parsuje `tourney_date`, sortuje chronologicznie i
oznacza `season` rokiem pliku. Robimy **jedno** wczytanie 2000..2025, a potem dzielimy:
- `historical` = sezony przed 2001 (czyli sezon 2000 -- tylko rozgrzewka cech, nie trafia do treningu),
- `span` = 2001..2025 (na nich liczymy cechy dynamiczne i ktore potem splitujemy)."""),

("code", """full = m.load_years(range(WARMUP_START, TEST_YEAR + 1), cols_base)
full = m.add_static_features(full, ROUND_ORDER)
historical = full[full["season"] < TRAIN_START].reset_index(drop=True)
span = full[full["season"] >= TRAIN_START].reset_index(drop=True)

print(f"Wczytano lacznie {len(full)} meczow ({WARMUP_START}-{TEST_YEAR})")
print(f"  rozgrzewka (<{TRAIN_START}): {len(historical)} meczow")
print(f"  span      (>={TRAIN_START}): {len(span)} meczow")

# Ile meczow per sezon w span (kontrola spojnosci danych)
print("\\nMecze per sezon (span):")
print(span["season"].value_counts().sort_index().to_string())"""),

("md", """## 3. Cechy dynamiczne na 2001-2025 (z rozgrzewka: sezon 2000)
`add_dynamic_features(span, historical)` liczy formy, H2H, surface form, statystyki serwisu itd. dla
kazdego meczu w `span`, korzystajac z historii (rozgrzewka 2000 + wczesniejsze mecze span). Funkcja
radzi sobie nawet z mala rozgrzewka. To najdluzszy krok obliczeniowy notebooka."""),

("code", """span_feat = add_dynamic_features(span, historical)   # funkcja z baseline (ns), nie z modulu m
print(f"Cechy dynamiczne policzone dla {len(span_feat)} meczow.")

# Podglad kilku kolumn cech dla najwczesniejszych meczow treningowych
sample_cols = [c for c in ["season", "winner_name", "loser_name",
                           "w_form", "l_form", "w_surface_form", "l_surface_form"]
               if c in span_feat.columns]
print(f"\\nPrzykladowe cechy (pierwsze mecze {TRAIN_START}):")
print(span_feat[span_feat["season"] == TRAIN_START][sample_cols].head(5).to_string(index=False))"""),

("md", """## 4. Label encoding (fit tylko na treningu) + split po sezonie
`surface` i `tourney_level` kodujemy `LabelEncoder`-em fitowanym **wylacznie** na meczach treningowych
(`season < 2024`). Nieznane kategorie w val/test mapujemy bezpiecznie na pierwsza znana klase (zero
wgladu w przyszlosc). Potem dzielimy `span_feat` na trzy roczniki i nadajemy `match_id`."""),

("code", """train_mask = span_feat["season"] < VAL_YEAR
le_surface, le_level = LabelEncoder(), LabelEncoder()
le_surface.fit(span_feat.loc[train_mask, "surface"])
le_level.fit(span_feat.loc[train_mask, "tourney_level"].astype(str))

def safe_transform(le, series):
    known = set(le.classes_)
    s = series.astype(str).where(series.astype(str).isin(known), le.classes_[0])
    return le.transform(s)

span_feat["surface_encoded"] = safe_transform(le_surface, span_feat["surface"])
span_feat["tourney_level_encoded"] = safe_transform(le_level, span_feat["tourney_level"].astype(str))

train_raw = span_feat[span_feat["season"] < VAL_YEAR].reset_index(drop=True)
val_raw   = span_feat[span_feat["season"] == VAL_YEAR].reset_index(drop=True)
test_raw  = span_feat[span_feat["season"] == TEST_YEAR].reset_index(drop=True)
for frame in (train_raw, val_raw, test_raw):
    frame["match_id"] = range(len(frame))

print(f"surface classes (fit na treningu): {list(le_surface.classes_)}")
print(f"tourney_level classes (fit na treningu): {list(le_level.classes_)}")
print(f"\\nMecze: train={len(train_raw)} ({TRAIN_START}-{VAL_YEAR-1})"
      f"  val={len(val_raw)} ({VAL_YEAR})  test={len(test_raw)} ({TEST_YEAR})")"""),

("md", """## 5. Symetryzacja -- ten sam matrix dla wszystkich modeli
Kazdy mecz zapisujemy z **dwoch** perspektyw (p1=zwyciezca / p1=przegrany), eliminujac arbitralny
labeling. Wersja `shuffle=False` (chronologiczna) sluzy do CV w `TimeSeriesSplit`, a `shuffle=True`
do finalnego fitu. Macierz `X_*[features]` jest **identyczna** dla RF, HGB i XGBoost -- jedyna roznica
miedzy modelami to algorytm."""),

("code", """train_cv  = symmetrize_data(train_raw, shuffle=False)
train_fit = symmetrize_data(train_raw, shuffle=True)
val_data  = symmetrize_data(val_raw, shuffle=True)
test_data = symmetrize_data(test_raw, shuffle=True)

X_tr_cv,  y_tr_cv  = train_cv[features],  train_cv["y"]
X_tr_fit, y_tr_fit = train_fit[features], train_fit["y"]
X_val,    y_val    = val_data[features],  val_data["y"]
X_test,   y_test   = test_data[features], test_data["y"]

print(f"Probki treningowe po symetryzacji: {len(X_tr_fit)} (2x meczow treningowych)")
print(f"Probki val: {len(X_val)}   |   probki test: {len(X_test)}")

# Sanity-check antysymetrii: ten sam mecz z dwoch perspektyw ma y=1 i y=0
mid = test_data["match_id"].iloc[0]
print("\\nTen sam mecz widziany z 2 perspektyw (kolumna y):")
print(test_data[test_data["match_id"] == mid][["match_id", "y", "p1_name", "p2_name"]].to_string(index=False))"""),

("md", """## 6. [1/3] Random Forest -- tuning + ewaluacja (baseline odniesienia)
`tune_and_eval` robi `RandomizedSearchCV` na chronologicznym CV (`neg_log_loss`), refituje najlepszy
model na pelnym treningu i zwraca match accuracy (val/test) oraz metryki kalibracji. RF jest naszym
punktem odniesienia -- to z nim porownujemy boosting."""),

("code", """rf_param_dist = {
    "n_estimators": [100, 200], "max_depth": [10, 20, None],
    "min_samples_leaf": [2, 5, 10], "max_features": ["sqrt", "log2"],
    "max_samples": [0.8, 1.0],
}
print("[1/3] Random Forest -- tuning (TimeSeriesSplit, neg_log_loss)...")
res_rf = m.tune_and_eval(
    "RandomForest",
    RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
    rf_param_dist, 8,
    X_tr_cv, y_tr_cv, X_tr_fit, y_tr_fit, X_val, y_val, val_data,
    X_test, y_test, test_data, compute_symmetric_match_evaluation, evaluate_calibration_quality)

print(f"  val_match ={res_rf['val_match']:.4f}   test_match={res_rf['test_match']:.4f}")
print(f"  Brier={res_rf['brier']:.4f}  logloss={res_rf['logloss']:.4f}  ECE={res_rf['ece']:.4f}")
print(f"  best_params: {res_rf['best_params']}")"""),

("md", """## 7. [2/3] HistGradientBoosting -- ten sam matrix, wiecej danych
Teraz boosting. Jesli "wiecej danych" mialo dac przewage boostingowi, to wlasnie tutaj (~128k probek)
powinno byc widac. Te same dane, ta sama metryka, inny algorytm."""),

("code", """hgb_param_dist = {
    "learning_rate": [0.03, 0.05, 0.1], "max_iter": [300, 500, 800],
    "max_leaf_nodes": [31, 63], "min_samples_leaf": [20, 50, 100],
    "l2_regularization": [0.0, 0.1, 1.0], "max_features": [0.6, 0.8, 1.0],
}
print("[2/3] HistGradientBoosting -- tuning...")
res_hgb = m.tune_and_eval(
    "HistGradBoost",
    HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=False),
    hgb_param_dist, 12,
    X_tr_cv, y_tr_cv, X_tr_fit, y_tr_fit, X_val, y_val, val_data,
    X_test, y_test, test_data, compute_symmetric_match_evaluation, evaluate_calibration_quality)

print(f"  val_match ={res_hgb['val_match']:.4f}   test_match={res_hgb['test_match']:.4f}")
print(f"  Brier={res_hgb['brier']:.4f}  logloss={res_hgb['logloss']:.4f}  ECE={res_hgb['ece']:.4f}")
print(f"  best_params: {res_hgb['best_params']}")"""),

("md", """## 8. [3/3] XGBoost -- trzeci zawodnik na tej samej macierzy
Drugi boosting (histogramowy, ale inna implementacja i regularyzacja). Pomijany automatycznie, gdyby
biblioteka byla niedostepna -- w tym srodowisku jest, wiec liczymy pelny tuning."""),

("code", """results = [res_rf, res_hgb]
if m.HAS_XGB:
    from xgboost import XGBClassifier
    xgb_param_dist = {
        "n_estimators": [300, 500, 800], "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9], "min_child_weight": [1, 5, 10],
    }
    print("[3/3] XGBoost -- tuning...")
    res_xgb = m.tune_and_eval(
        "XGBoost",
        XGBClassifier(tree_method="hist", objective="binary:logistic",
                      eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
        xgb_param_dist, 12,
        X_tr_cv, y_tr_cv, X_tr_fit, y_tr_fit, X_val, y_val, val_data,
        X_test, y_test, test_data, compute_symmetric_match_evaluation, evaluate_calibration_quality)
    results.append(res_xgb)
    print(f"  val_match ={res_xgb['val_match']:.4f}   test_match={res_xgb['test_match']:.4f}")
    print(f"  Brier={res_xgb['brier']:.4f}  logloss={res_xgb['logloss']:.4f}  ECE={res_xgb['ece']:.4f}")
    print(f"  best_params: {res_xgb['best_params']}")
else:
    print("[3/3] XGBoost POMINIETY (brak biblioteki).")"""),

("md", """## 9. Tabela porownawcza + delty vs Random Forest
Zestawiamy trzy modele: match accuracy na val i test (caly sezon 2025) oraz jakosc kalibracji
(Brier, log-loss, ECE). Delty liczymy wzgledem RF -- chcemy zobaczyc, czy ktorykolwiek boosting
**pobil RF na accuracy**, i czy lepsza kalibracja (jesli jest) przeklada sie na cokolwiek poza
jakoscia prawdopodobienstw."""),

("code", """comp = pd.DataFrame([{
    "model": r["name"], "val_match": r["val_match"], "test_match": r["test_match"],
    "Brier": r["brier"], "logloss": r["logloss"], "ECE": r["ece"],
} for r in results])
print("=" * 78)
print(f"WIELO-SEZONOWY TRENING ({TRAIN_START}-{VAL_YEAR-1}, ~{len(X_tr_fit)} probek) | test {TEST_YEAR}")
print("=" * 78)
print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

rf = next(r for r in results if r["name"] == "RandomForest")
print("\\nDelty vs Random Forest:")
for r in results:
    if r["name"] != "RandomForest":
        print(f"  {r['name']:<14} test_match={r['test_match']-rf['test_match']:+.4f}   "
              f"Brier={r['brier']-rf['brier']:+.4f}   logloss={r['logloss']-rf['logloss']:+.4f}")
print(f"\\nUWAGA: test = caly sezon {TEST_YEAR} (~{len(test_raw)} meczow). CI ~ +/-2 p.p.")"""),

("md", """## Wnioski

**Konfiguracja:** trening **2001-2023** (~123 tys. probek po symetryzacji = 61 556 meczow; sezon 2000
= rozgrzewka cech), walidacja 2024, test = **caly sezon 2025** (2647 meczow).

**Wyniki (test match accuracy -- patrz tabela w sekcji 9):**

| model | test match acc | delta vs RF | Brier |
|---|---|---|---|
| XGBoost | **0.6490** | **+0.0030** | 0.2165 |
| Random Forest | **0.6460** | -- | 0.2182 |
| HistGradientBoosting | **0.6411** | **-0.0049** | 0.2172 |

Rozpietosc miedzy modelami to ok. **0.8 p.p.** -- czyli **wszystkie w granicach szumu** (CI ~ +/-2 p.p.
przy 2647 meczach). Zaden algorytm nie wygrywa w sposob wiarygodny.

**Najwazniejsze:** zwiekszenie zbioru treningowego ~36x (z ~3,5 tys. do ~123 tys. probek) **NIE
przebilo sufitu ~65% dla ZADNEGO algorytmu**. To potwierdza, ze sciana lezy w **cechach i samym
problemie**, a nie w algorytmie ani ilosci danych.

**Niuans:** na tym najwiekszym zbiorze XGBoost jest minimalnie z przodu -- i na accuracy (+0.30 p.p.),
i na kalibracji (najnizszy Brier) -- a HistGradientBoosting najgorszy. Spojne z intuicja, ze boosting
rozwija sie z iloscia danych (na ~72 tys. probek RF i XGBoost remisowaly po 0.6494; na ~123 tys.
XGBoost lekko wyprzedza). Ale zysk **pozostaje w granicach szumu** -- to nie jest wiarygodna przewaga.

**Random Forest jako model glowny** to wybor **praktyczny** (stabilny, bez dodatkowej zaleznosci,
remisuje lub jest bliski najlepszemu w kazdej konfiguracji), a **nie** dlatego, ze jest mierzalnie
najlepszy.

**Zastrzezenia:** to **JEDEN** trening wielo-sezonowy (NIE walk-forward); jego baseline (~0.646 na
2647 meczach) liczy sie na **innych danych** niz single-season 0.6566 -- nie porownywac 1:1."""),
]

make_and_run("TPM_Experiment_MultiSeason.ipynb", cells, timeout=7200)
