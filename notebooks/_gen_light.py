"""Generuje i wykonuje 4 lekkie notebooki: fatigue, enriched, ewma, hgb."""
from _nbtools import make_and_run

IMPORTS = """import sys, io, contextlib, runpy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
sys.path.insert(0, str(Path("../src").resolve()))"""

BASELINE = """BASE = Path("../src/tennis_model.py").resolve()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(BASE))
symmetrize_data = ns["symmetrize_data"]
compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
evaluate_calibration_quality = ns["evaluate_calibration_quality"]
baseline_search = ns["search"]
RANDOM_STATE = ns["RANDOM_STATE"]
base_features = list(ns["features"]); cols_base = list(ns["cols_base"])
df_train_raw = ns["df_train_raw"].copy(); df_val_raw = ns["df_val_raw"].copy(); df_test_raw = ns["df_test_raw"].copy()
baseline_val_acc = float(ns["val_acc"]); baseline_test_acc = float(ns["test_acc"]); baseline_match_acc = float(ns["match_accuracy"])
print(f"Baseline: val={baseline_val_acc:.4f}  test={baseline_test_acc:.4f}  match={baseline_match_acc:.4f}  (cech: {len(base_features)})")"""

LOAD_TARGET = """full_target = pd.read_csv(data_file(TARGET_YEAR))
full_target["tourney_date"] = pd.to_datetime(full_target["tourney_date"], format="%Y%m%d")
full_target = full_target.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
full_target_base = full_target[cols_base + ["tourney_id", "minutes"]].dropna(subset=cols_base).reset_index(drop=True)
n_train, n_val, n_test = len(df_train_raw), len(df_val_raw), len(df_test_raw)
assert len(full_target_base) == n_train + n_val + n_test"""

# ===================== FATIGUE =====================
fatigue_cells = [
("md", """# Eksperyment: Cechy zmęczenia / Fatigue (Sprint 3b)

## Cel
Czy uwzględnienie zmęczenia gracza poprawia predykcje? Dwie nowe cechy (liczone bez leakage,
z chronologicznego indeksu): **rest_days** (dni od ostatniego meczu, cap 60) oraz
**tourney_minutes** (minuty zagrane w bieżącym turnieju -- skumulowane przez wcześniejsze rundy).
Symetryzowane do p1/p2 + różnice. Te same tuned HP co baseline (ablation)."""),
("code", IMPORTS + """
from tennis_model_fatigue import compute_fatigue_for_2024, NEW_FEATURES, data_file, TARGET_YEAR, HISTORY_START_YEAR
print("Rok docelowy:", TARGET_YEAR, "| nowe cechy:", NEW_FEATURES)"""),
("md", "## 1. Reuse baseline pipeline"),
("code", BASELINE),
("md", """## 2. Liczenie cech zmęczenia (leakage-safe)
`compute_fatigue_for_2024` przetwarza historię + rok docelowy chronologicznie i dla każdego meczu
liczy rest_days oraz tourney_minutes ze ŚCIŚLE wcześniejszych meczów (indeks + bisect)."""),
("code", LOAD_TARGET + """
fatigue = compute_fatigue_for_2024(full_target_base)
fat_train = fatigue.iloc[:n_train].reset_index(drop=True)
fat_val = fatigue.iloc[n_train:n_train + n_val].reset_index(drop=True)
fat_test = fatigue.iloc[n_train + n_val:].reset_index(drop=True)
print(fatigue.describe().round(1).to_string())"""),
("md", "## 3. Doklejenie + symetryzacja (w_/l_ -> p1_/p2_) + różnice"),
("code", """def attach(df_raw, fat):
    df_raw = df_raw.copy().reset_index(drop=True)
    for col in ("w_rest_days", "l_rest_days", "w_tourney_minutes", "l_tourney_minutes"):
        df_raw[col] = fat[col].to_numpy()
    return df_raw
df_train_raw = attach(df_train_raw, fat_train); df_val_raw = attach(df_val_raw, fat_val); df_test_raw = attach(df_test_raw, fat_test)

def build_split(df_raw, shuffle):
    sym = symmetrize_data(df_raw, shuffle=shuffle)
    raw_map = df_raw[["match_id", "w_rest_days", "l_rest_days", "w_tourney_minutes", "l_tourney_minutes"]]
    sym = sym.merge(raw_map, on="match_id", how="left", validate="many_to_one")
    w = (sym["y"] == 1).to_numpy()
    sym["p1_rest_days"] = np.where(w, sym["w_rest_days"], sym["l_rest_days"])
    sym["p2_rest_days"] = np.where(w, sym["l_rest_days"], sym["w_rest_days"])
    sym["p1_tourney_minutes"] = np.where(w, sym["w_tourney_minutes"], sym["l_tourney_minutes"])
    sym["p2_tourney_minutes"] = np.where(w, sym["l_tourney_minutes"], sym["w_tourney_minutes"])
    sym["rest_days_diff"] = sym["p1_rest_days"] - sym["p2_rest_days"]
    sym["tourney_minutes_diff"] = sym["p1_tourney_minutes"] - sym["p2_tourney_minutes"]
    return sym
train_data = build_split(df_train_raw, True); val_data = build_split(df_val_raw, True); test_data = build_split(df_test_raw, True)
features = base_features + NEW_FEATURES
print("Cech razem:", len(features))"""),
("md", "## 4. Trening RF + ewaluacja"),
("code", """X_train, y_train = train_data[features], train_data["y"]
X_val, y_val = val_data[features], val_data["y"]; X_test, y_test = test_data[features], test_data["y"]
best_rf = RandomForestClassifier(**baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE)
best_rf.fit(X_train, y_train)
val_acc = float(accuracy_score(y_val, best_rf.predict(X_val))); test_acc = float(accuracy_score(y_test, best_rf.predict(X_test)))
proba = best_rf.predict_proba(X_test)[:, 1]; test_data["p1_win_probability"] = proba
_, match_acc = compute_symmetric_match_evaluation(test_data); q = evaluate_calibration_quality(y_test.to_numpy(), proba)
imp = pd.DataFrame({"feature": features, "importance": best_rf.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True); imp["rank"] = imp.index + 1
print(f"baseline       match={baseline_match_acc:.4f}")
print(f"+fatigue       match={match_acc:.4f}  Brier={q['brier_score']:.4f}  DELTA={match_acc-baseline_match_acc:+.4f}")
for f in NEW_FEATURES:
    r = imp[imp.feature == f].iloc[0]; print(f"  {f:<22} rank {int(r['rank']):>2}/{len(features)}")"""),
("md", """## Wnioski
Cechy zmęczenia (dni odpoczynku, minuty na korcie) praktycznie nie ruszają trafności. Na walidacji przez 6 sezonów wychodzi +0,03 p.p. (McNemar p = 1,0). Model z nich korzysta, ale przewaga się nie utrzymuje — mieści się w szumie."""),
]

# ===================== ENRICHED =====================
enriched_cells = [
("md", """# Eksperyment: Model zbiorczy / Enriched (Sprint 3d)

## Cel
Połączyć dwa wygrywające (na pojedynczym teście) zestawy cech: **surface_speed (3) + fatigue (6)**.
Pytanie: czy zyski się sumują, czy uderza nadmiar cech (curse of dimensionality)?"""),
("code", IMPORTS + """
from tennis_model_surface_speed import build_court_pace_lookup, court_pace_index
from tennis_model_fatigue import compute_fatigue_for_2024
from tennis_model_enriched import SPEED_FEATURES, FATIGUE_FEATURES, NEW_FEATURES, data_file, TARGET_YEAR
print("Nowe cechy:", len(NEW_FEATURES), "=", len(SPEED_FEATURES), "speed +", len(FATIGUE_FEATURES), "fatigue")"""),
("md", "## 1. Reuse baseline pipeline"),
("code", BASELINE),
("md", "## 2. Budowa kontekstu: court_pace + fatigue (wyrównane do roku docelowego)"),
("code", LOAD_TARGET + """
lookup = build_court_pace_lookup()
cpi = np.array([court_pace_index(t, s, lookup) for t, s in zip(full_target_base["tourney_id"], full_target_base["surface"])])
fat = compute_fatigue_for_2024(full_target_base)
context = pd.DataFrame({"court_pace_index": cpi,
    "w_rest_days": fat["w_rest_days"].to_numpy(), "l_rest_days": fat["l_rest_days"].to_numpy(),
    "w_tourney_minutes": fat["w_tourney_minutes"].to_numpy(), "l_tourney_minutes": fat["l_tourney_minutes"].to_numpy()})
ctx_train = context.iloc[:n_train].reset_index(drop=True)
ctx_val = context.iloc[n_train:n_train+n_val].reset_index(drop=True)
ctx_test = context.iloc[n_train+n_val:].reset_index(drop=True)
print("Kontekst:", list(context.columns))"""),
("md", "## 3. Doklejenie + symetryzacja + interakcje serve x speed + fatigue p1/p2"),
("code", """def attach(df_raw, ctx):
    df_raw = df_raw.copy().reset_index(drop=True)
    for col in context.columns: df_raw[col] = ctx[col].to_numpy()
    return df_raw
df_train_raw = attach(df_train_raw, ctx_train); df_val_raw = attach(df_val_raw, ctx_val); df_test_raw = attach(df_test_raw, ctx_test)
raw_ctx = ["match_id"] + list(context.columns)
def build_split(df_raw, shuffle):
    sym = symmetrize_data(df_raw, shuffle=shuffle).merge(df_raw[raw_ctx], on="match_id", how="left", validate="many_to_one")
    w = (sym["y"] == 1).to_numpy()
    sym["ace_speed_diff"] = (sym["p1_ace_rate"]-sym["p2_ace_rate"])*sym["court_pace_index"]
    sym["first_won_speed_diff"] = (sym["p1_first_won_pct"]-sym["p2_first_won_pct"])*sym["court_pace_index"]
    sym["p1_rest_days"]=np.where(w,sym["w_rest_days"],sym["l_rest_days"]); sym["p2_rest_days"]=np.where(w,sym["l_rest_days"],sym["w_rest_days"])
    sym["p1_tourney_minutes"]=np.where(w,sym["w_tourney_minutes"],sym["l_tourney_minutes"]); sym["p2_tourney_minutes"]=np.where(w,sym["l_tourney_minutes"],sym["w_tourney_minutes"])
    sym["rest_days_diff"]=sym["p1_rest_days"]-sym["p2_rest_days"]; sym["tourney_minutes_diff"]=sym["p1_tourney_minutes"]-sym["p2_tourney_minutes"]
    return sym
train_data=build_split(df_train_raw,True); test_data=build_split(df_test_raw,True); val_data=build_split(df_val_raw,True)
features = base_features + NEW_FEATURES
print("Cech razem:", len(features))"""),
("md", "## 4. Trening RF + ewaluacja"),
("code", """X_train,y_train=train_data[features],train_data["y"]; X_val,y_val=val_data[features],val_data["y"]; X_test,y_test=test_data[features],test_data["y"]
best_rf = RandomForestClassifier(**baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE).fit(X_train,y_train)
val_acc=float(accuracy_score(y_val,best_rf.predict(X_val))); test_acc=float(accuracy_score(y_test,best_rf.predict(X_test)))
proba=best_rf.predict_proba(X_test)[:,1]; test_data["p1_win_probability"]=proba
_,match_acc=compute_symmetric_match_evaluation(test_data); q=evaluate_calibration_quality(y_test.to_numpy(),proba)
print(f"baseline           match={baseline_match_acc:.4f}")
print(f"+speed+fatigue     match={match_acc:.4f}  Brier={q['brier_score']:.4f}  DELTA={match_acc-baseline_match_acc:+.4f}")"""),
("md", """## Wnioski
Połączenie prędkości kortu i zmęczenia (9 cech naraz) nie sumuje się do realnego zysku. Na walidacji przez 6 sezonów wychodzi +0,20 p.p. (McNemar p = 0,66), czyli nieistotnie. Dobra wiadomość jest taka, że dołożenie tylu cech niczego nie psuje — ale też nic nie wnosi. Znów ten sam sufit ~65%."""),
]

# ===================== EWMA =====================
ewma_cells = [
("md", """# Eksperyment: EWMA / recency weighting (Sprint 3c)

## Cel
Zamiast prostej średniej z 10 ostatnich meczów (SMA) użyć **wykładniczego ważenia** -- starsze mecze
maleją gładko (alpha=0.18, half-life ~3.5 meczu). Nadpisujemy cechy formy i serwisu (te same nazwy),
nie dodajemy nowych. To zmiana REPREZENTACJI cech."""),
("code", IMPORTS + """
from tennis_model_ewma_ablation import compute_ewma_features, OVERWRITE_COLS, ALPHA, data_file, TARGET_YEAR, HISTORY_START_YEAR
print("alpha:", ALPHA, "| nadpisywane kolumny:", len(OVERWRITE_COLS))"""),
("md", "## 1. Reuse baseline pipeline"),
("code", BASELINE + """
features = base_features"""),
("md", """## 2. Liczenie cech EWMA z chronologii (leakage-safe)
`compute_ewma_features` przetwarza historię + rok docelowy incrementalnie, utrzymując stan EWMA
per gracz, i zapisuje pre-match wartości (forma/serwis/surface_form)."""),
("code", """full_target = pd.read_csv(data_file(TARGET_YEAR))
full_target["tourney_date"] = pd.to_datetime(full_target["tourney_date"], format="%Y%m%d")
full_target = full_target.sort_values(["tourney_date","match_num"]).reset_index(drop=True)
full_target_base = full_target[cols_base].dropna(subset=cols_base).reset_index(drop=True)
n_train,n_val,n_test = len(df_train_raw),len(df_val_raw),len(df_test_raw)
assert len(full_target_base)==n_train+n_val+n_test
ewma = compute_ewma_features(full_target_base, cols_base)
e_train=ewma.iloc[:n_train].reset_index(drop=True); e_val=ewma.iloc[n_train:n_train+n_val].reset_index(drop=True); e_test=ewma.iloc[n_train+n_val:].reset_index(drop=True)
print("Kolumny EWMA:", OVERWRITE_COLS[:4], "...")"""),
("md", "## 3. Nadpisanie cech SMA -> EWMA, symetryzacja, retrening"),
("code", """def overwrite(df_raw, e):
    df_raw = df_raw.copy().reset_index(drop=True)
    for col in OVERWRITE_COLS: df_raw[col] = e[col].to_numpy()
    return df_raw
df_train_raw=overwrite(df_train_raw,e_train); df_val_raw=overwrite(df_val_raw,e_val); df_test_raw=overwrite(df_test_raw,e_test)
train_data=symmetrize_data(df_train_raw,shuffle=True); val_data=symmetrize_data(df_val_raw,shuffle=True); test_data=symmetrize_data(df_test_raw,shuffle=True)
X_train,y_train=train_data[features],train_data["y"]; X_val,y_val=val_data[features],val_data["y"]; X_test,y_test=test_data[features],test_data["y"]
best_rf = RandomForestClassifier(**baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE).fit(X_train,y_train)
val_acc=float(accuracy_score(y_val,best_rf.predict(X_val))); test_acc=float(accuracy_score(y_test,best_rf.predict(X_test)))
proba=best_rf.predict_proba(X_test)[:,1]; test_data["p1_win_probability"]=proba
_,match_acc=compute_symmetric_match_evaluation(test_data); q=evaluate_calibration_quality(y_test.to_numpy(),proba)
print(f"baseline (SMA)  val={baseline_val_acc:.4f}  test={baseline_test_acc:.4f}  match={baseline_match_acc:.4f}")
print(f"EWMA            val={val_acc:.4f}  test={test_acc:.4f}  match={match_acc:.4f}  Brier={q['brier_score']:.4f}")
print(f"DELTA match: {match_acc-baseline_match_acc:+.4f}")"""),
("md", """## Wnioski
EWMA dało niespójny wynik — walidacja czasem rośnie, ale na teście i match accuracy prawie nic się nie zmienia. Powód jest prosty: baseline ma już okno 365 dni, które i tak wyłapuje większość tego, co daje ważenie świeższych meczów. Czyli ważenie wykładnicze nie poprawia modelu w istotny sposób."""),
]

# ===================== HGB =====================
hgb_cells = [
("md", """# Eksperyment: HistGradientBoosting vs Random Forest (Sprint 2)

## Cel
Sprawdzić, czy gradient boosting (HistGradientBoosting z sklearn) pobije Random Forest na tych
samych cechach i danych. Dwa warianty HGB: numeryczny (jak RF) oraz z **natywnymi kategoriami**
(surface/tourney_level jako nominalne -- główna przewaga HGB)."""),
("code", IMPORTS + """
from sklearn.ensemble import HistGradientBoostingClassifier
from tennis_model_hgb import tune_and_eval_hgb, print_row"""),
("md", """## 1. Reuse baseline pipeline (RF już policzony)
Baseline zwraca gotowy RF + dane CV i test. Liczymy jakość prawdopodobieństw RF do porównania."""),
("code", BASELINE + """
features = base_features
X_train_cv, y_train_cv = ns["X_train_cv"], ns["y_train_cv"]
X_val, y_val = ns["X_val"], ns["y_val"]; X_test, y_test = ns["X_test"], ns["y_test"]
test_data = ns["test_data"]
best_rf = ns["best_rf"]
rf_proba = best_rf.predict_proba(X_test)[:, 1]
rf_q = evaluate_calibration_quality(y_test.to_numpy(), rf_proba)
print(f"RF: match={baseline_match_acc:.4f}  Brier={rf_q['brier_score']:.4f}")"""),
("md", """## 2. Strojenie HGB (RandomizedSearchCV, neg_log_loss) -- dwa warianty
`tune_and_eval_hgb` stroi HGB na TimeSeriesSplit i ocenia val/test/match + kalibrację."""),
("code", """common = dict(features=features, X_train_cv=X_train_cv, y_train_cv=y_train_cv, df_train_raw=df_train_raw,
    symmetrize_data=symmetrize_data, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, test_data=test_data,
    compute_symmetric_match_evaluation=compute_symmetric_match_evaluation,
    evaluate_calibration_quality=evaluate_calibration_quality, RANDOM_STATE=RANDOM_STATE)
hgb_num = tune_and_eval_hgb(label="HGB (numeric)", categorical_features=None, **common)
cat_cols = [c for c in ("surface", "tourney_level") if c in features]
hgb_cat = tune_and_eval_hgb(label="HGB (kategorie)", categorical_features=cat_cols, **common)
print("gotowe")"""),
("md", "## 3. Porównanie RF vs HGB"),
("code", """print(f"{'model':<16}{'val':>9}{'test':>9}{'match':>9}{'Brier':>9}{'logloss':>9}")
print(f"{'RandomForest':<16}{baseline_val_acc:>9.4f}{baseline_test_acc:>9.4f}{baseline_match_acc:>9.4f}{rf_q['brier_score']:>9.4f}{rf_q['log_loss']:>9.4f}")
for r in (hgb_num, hgb_cat):
    print(f"{r['label']:<16}{r['val_acc']:>9.4f}{r['test_acc']:>9.4f}{r['match_acc']:>9.4f}{r['brier']:>9.4f}{r['logloss']:>9.4f}")
for r in (hgb_num, hgb_cat):
    print(f"DELTA ({r['label']} - RF): match={r['match_acc']-baseline_match_acc:+.4f}")"""),
("md", """## Wnioski
HistGradientBoosting wyszedł mniej więcej na równi z Random Forest — raz odrobinę gorzej, raz odrobinę lepiej, wszystko w granicach szumu — a kalibrację miał słabszą. Przy około 3500 meczach treningowych boosting nie ma z czego rozwinąć przewagi i wybiera mocno regularyzowane ustawienia. Dlatego zostaję przy Random Forest. Ten sam test powtórzyłem na dużo większym zbiorze (notebook multiseason, ~123 tys. próbek) — wynik dalej ~65%."""),
]

for name, cells in [
    ("TPM_Experiment_Fatigue.ipynb", fatigue_cells),
    ("TPM_Experiment_Enriched.ipynb", enriched_cells),
    ("TPM_Experiment_EWMA.ipynb", ewma_cells),
    ("TPM_Experiment_HGB.ipynb", hgb_cells),
]:
    try:
        make_and_run(name, cells, timeout=1800)
    except Exception as e:
        print(f"[BŁĄD] {name}: {type(e).__name__}: {e}")
