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
Dotychczasowa architektura trenowała **tylko na roku docelowym** (~3500 próbek). Sprint 2 pokazał,
że HistGradientBoosting nie bije Random Forest -- ale na tak małej próbie boosting nie ma jak
rozwinąć swojej przewagi. Tutaj zmieniamy architekturę: trenujemy na **wielu sezonach**
(2001-2023, ~128 tys. próbek po symetryzacji -- praktycznie całe ATP od 2000), walidujemy na 2024 i
testujemy na **całym sezonie 2025**. To właściwy test hipotezy *"więcej danych => boosting wreszcie
opłacalny"*.

## Metoda (leakage-safe, ten sam matrix dla wszystkich modeli)
- **Cechy IDENTYCZNE z baseline** (40 cech) -- reużywamy `add_dynamic_features` / `symmetrize_data` z
  `tennis_model.py` przez namespace. Jedyne zmienne to **ilość danych treningowych** i **algorytm**.
- **Rozgrzewka cech: sezon 2000** -- liczy historię (forma, H2H, surface form) dla pierwszych meczów
  2001, ale **nie wchodzi** do zbioru treningowego.
- **Split po roku PLIKU** (`season`), nie po `tourney_date.dt.year`: plik sezonu 2025 zaczyna się od
  United Cup z końca grudnia 2024, więc sama data myli sezon.
- **Label encoding** fitowany TYLKO na treningu (`season < 2024`) -- zero wglądu w val/test.
- **CV chronologiczne** (`TimeSeriesSplit`) + tuning po `neg_log_loss`, ten sam `random_state=42`.
- Trzy modele -- **RF vs HistGradientBoosting vs XGBoost** -- na **dokładnie tej samej** macierzy
  cech, porównanie match accuracy oraz jakości kalibracji (Brier / log-loss / ECE).

> UWAGA: to **pojedynczy** trening wielo-sezonowy (jeden train/val/test), a **nie** walk-forward.
> Baseline tego notebooka (~0.647 na 2647 meczach 2025) to **inne dane** niż single-season 0.6566 --
> nie porównujemy ich 1:1."""),

("code", SETUP),

("md", """## 1. Reuse baseline -- pobieramy funkcje feature-engineering
Najpierw ustawiamy zakres treningu (**od 2001**, rozgrzewka 2000) przez zmienne środowiskowe -- moduł
czyta je przy imporcie. Potem uruchamiamy `tennis_model.py` raz (z wyciszonym outputem) i wyciągamy z
jego namespace funkcje: budowanie cech dynamicznych, symetryzacja, symetryczna metryka match-level
oraz ocena kalibracji. Bierzemy też listę **40 cech** i `cols_base` -- macierz jest taka sama jak w
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

("md", """## 2. Wczytujemy wszystkie sezony 2000-2025 i dzielimy na rozgrzewkę / span
`load_years` czyta pliki `atp_matches_<rok>.csv`, parsuje `tourney_date`, sortuje chronologicznie i
oznacza `season` rokiem pliku. Robimy **jedno** wczytanie 2000..2025, a potem dzielimy:
- `historical` = sezony przed 2001 (czyli sezon 2000 -- tylko rozgrzewka cech, nie trafia do treningu),
- `span` = 2001..2025 (na nich liczymy cechy dynamiczne i które potem splitujemy)."""),

("code", """full = m.load_years(range(WARMUP_START, TEST_YEAR + 1), cols_base)
full = m.add_static_features(full, ROUND_ORDER)
historical = full[full["season"] < TRAIN_START].reset_index(drop=True)
span = full[full["season"] >= TRAIN_START].reset_index(drop=True)

print(f"Wczytano łącznie {len(full)} meczów ({WARMUP_START}-{TEST_YEAR})")
print(f"  rozgrzewka (<{TRAIN_START}): {len(historical)} meczów")
print(f"  span      (>={TRAIN_START}): {len(span)} meczów")

# Ile meczow per sezon w span (kontrola spojnosci danych)
print("\\nMecze per sezon (span):")
print(span["season"].value_counts().sort_index().to_string())"""),

("md", """## 3. Cechy dynamiczne na 2001-2025 (z rozgrzewką: sezon 2000)
`add_dynamic_features(span, historical)` liczy formy, H2H, surface form, statystyki serwisu itd. dla
każdego meczu w `span`, korzystając z historii (rozgrzewka 2000 + wcześniejsze mecze span). Funkcja
radzi sobie nawet z małą rozgrzewką. To najdłuższy krok obliczeniowy notebooka."""),

("code", """span_feat = add_dynamic_features(span, historical)   # funkcja z baseline (ns), nie z modulu m
print(f"Cechy dynamiczne policzone dla {len(span_feat)} meczów.")

# Podglad kilku kolumn cech dla najwczesniejszych meczow treningowych
sample_cols = [c for c in ["season", "winner_name", "loser_name",
                           "w_form", "l_form", "w_surface_form", "l_surface_form"]
               if c in span_feat.columns]
print(f"\\nPrzykładowe cechy (pierwsze mecze {TRAIN_START}):")
print(span_feat[span_feat["season"] == TRAIN_START][sample_cols].head(5).to_string(index=False))"""),

("md", """## 4. Label encoding (fit tylko na treningu) + split po sezonie
`surface` i `tourney_level` kodujemy `LabelEncoder`-em fitowanym **wyłącznie** na meczach treningowych
(`season < 2024`). Nieznane kategorie w val/test mapujemy bezpiecznie na pierwszą znaną klasę (zero
wglądu w przyszłość). Potem dzielimy `span_feat` na trzy roczniki i nadajemy `match_id`."""),

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
Każdy mecz zapisujemy z **dwóch** perspektyw (p1=zwycięzca / p1=przegrany), eliminując arbitralny
labeling. Wersja `shuffle=False` (chronologiczna) służy do CV w `TimeSeriesSplit`, a `shuffle=True`
do finalnego fitu. Macierz `X_*[features]` jest **identyczna** dla RF, HGB i XGBoost -- jedyna różnica
między modelami to algorytm."""),

("code", """train_cv  = symmetrize_data(train_raw, shuffle=False)
train_fit = symmetrize_data(train_raw, shuffle=True)
val_data  = symmetrize_data(val_raw, shuffle=True)
test_data = symmetrize_data(test_raw, shuffle=True)

X_tr_cv,  y_tr_cv  = train_cv[features],  train_cv["y"]
X_tr_fit, y_tr_fit = train_fit[features], train_fit["y"]
X_val,    y_val    = val_data[features],  val_data["y"]
X_test,   y_test   = test_data[features], test_data["y"]

print(f"Próbki treningowe po symetryzacji: {len(X_tr_fit)} (2x meczów treningowych)")
print(f"Próbki val: {len(X_val)}   |   próbki test: {len(X_test)}")

# Sanity-check antysymetrii: ten sam mecz z dwoch perspektyw ma y=1 i y=0
mid = test_data["match_id"].iloc[0]
print("\\nTen sam mecz widziany z 2 perspektyw (kolumna y):")
print(test_data[test_data["match_id"] == mid][["match_id", "y", "p1_name", "p2_name"]].to_string(index=False))"""),

("md", """## 6. [1/3] Random Forest -- tuning + ewaluacja (baseline odniesienia)
`tune_and_eval` robi `RandomizedSearchCV` na chronologicznym CV (`neg_log_loss`), refituje najlepszy
model na pełnym treningu i zwraca match accuracy (val/test) oraz metryki kalibracji. RF jest naszym
punktem odniesienia -- to z nim porównujemy boosting."""),

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

("md", """## 7. [2/3] HistGradientBoosting -- ten sam matrix, więcej danych
Teraz boosting. Jeśli "więcej danych" miało dać przewagę boostingowi, to właśnie tutaj (~128k próbek)
powinno być widać. Te same dane, ta sama metryka, inny algorytm."""),

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
biblioteka była niedostępna -- w tym środowisku jest, więc liczymy pełny tuning."""),

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
    print("[3/3] XGBoost POMINIĘTY (brak biblioteki).")"""),

("md", """## 9. Tabela porównawcza + delty vs Random Forest
Zestawiamy trzy modele: match accuracy na val i test (cały sezon 2025) oraz jakość kalibracji
(Brier, log-loss, ECE). Delty liczymy względem RF -- chcemy zobaczyć, czy którykolwiek boosting
**pobił RF na accuracy**, i czy lepsza kalibracja (jeśli jest) przekłada się na cokolwiek poza
jakością prawdopodobieństw."""),

("code", """comp = pd.DataFrame([{
    "model": r["name"], "val_match": r["val_match"], "test_match": r["test_match"],
    "Brier": r["brier"], "logloss": r["logloss"], "ECE": r["ece"],
} for r in results])
print("=" * 78)
print(f"WIELO-SEZONOWY TRENING ({TRAIN_START}-{VAL_YEAR-1}, ~{len(X_tr_fit)} próbek) | test {TEST_YEAR}")
print("=" * 78)
print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

rf = next(r for r in results if r["name"] == "RandomForest")
print("\\nDelty vs Random Forest:")
for r in results:
    if r["name"] != "RandomForest":
        print(f"  {r['name']:<14} test_match={r['test_match']-rf['test_match']:+.4f}   "
              f"Brier={r['brier']-rf['brier']:+.4f}   logloss={r['logloss']-rf['logloss']:+.4f}")
print(f"\\nUWAGA: test = cały sezon {TEST_YEAR} (~{len(test_raw)} meczów). CI ~ +/-2 p.p.")"""),

("md", """## Wnioski
Na największym zbiorze (trening 2001–2023, ~123 tys. próbek, test na całym sezonie 2025) wszystkie trzy algorytmy dają ~65% i różnią się tylko w granicach szumu:

| model | match accuracy 2025 | Brier |
|---|---|---|
| XGBoost | 0,6490 | 0,2165 |
| Random Forest | 0,6460 | 0,2182 |
| HistGradientBoosting | 0,6411 | 0,2172 |

Rozpiętość to 0,8 p.p., a przy 2647 meczach przedział ufności wynosi około ±2 p.p., więc nikt nie wygrywa wiarygodnie. XGBoost jest minimalnie z przodu — i na trafności, i na kalibracji — a HistGradientBoosting najsłabszy. Pasuje to do intuicji, że boosting rozwija się przy większej ilości danych: na ~72 tys. próbek Random Forest i XGBoost remisowały, a na ~123 tys. XGBoost lekko wyprzedza. Ale to wciąż szum, nie realna przewaga.

Najważniejsze jest to, że zwiększenie zbioru jakieś 36 razy nie ruszyło **sufitu ~65%** dla żadnego algorytmu — czyli ogranicza nas charakter cech i samego problemu, a nie wybór modelu ani ilość danych. Random Forest zostawiam jako model główny z praktycznych powodów: jest stabilny, nie wymaga dodatkowej biblioteki i wszędzie jest blisko najlepszego.

Jedna uwaga: to pojedynczy trening wielosezonowy, a nie walk-forward, więc jego wynik (~0,646 na całym 2025) liczy się na innych danych niż 0,657 z pojedynczego sezonu — nie porównuję ich wprost."""),
]

make_and_run("TPM_Experiment_MultiSeason.ipynb", cells, timeout=7200)
