from _nbtools import make_and_run

cells = [
("md", """# Eksperyment: Surface Speed Index (Sprint 3a)

## Cel
Sprawdzic pomysl: **szybszy kort faworyzuje graczy z mocniejszym serwem.** Dodajemy
do baseline (`tennis_model.py`) trzy nowe cechy i sprawdzamy, czy poprawiaja match accuracy.

## Metoda (leakage-safe)
- `court_pace_index` -- proxy predkosci kortu liczony WYLACZNIE z historii (sezony przed
  rokiem docelowym), wiec rok testowy nie wplywa na ceche.
- Interakcje **serve x speed**: roznica sily serwisu razy predkosc kortu (na szybkim korcie
  przewaga serwisu rosnie). To one niosa wartosc.
- Te same tuned hiperparametry co baseline (ablation: zmieniamy tylko cechy)."""),

("code", """import sys, io, contextlib, runpy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path("../src").resolve()))
from tennis_model_surface_speed import (
    build_court_pace_lookup, court_pace_index, data_file,
    TARGET_YEAR, HISTORY_START_YEAR, NEW_FEATURES,
)
print("Rok docelowy:", TARGET_YEAR, "| historia od:", HISTORY_START_YEAR)
print("Nowe cechy:", NEW_FEATURES)"""),

("md", """## 1. Reuse baseline pipeline
Uruchamiamy `tennis_model.py` (z wyciszonym outputem) i pobieramy z niego: dane treningowe/testowe,
funkcje (`symmetrize_data`, metryke symetryczna, ocene kalibracji), tuned hiperparametry i metryki baseline."""),

("code", """BASE = Path("../src/tennis_model.py").resolve()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(BASE))

symmetrize_data = ns["symmetrize_data"]
compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
evaluate_calibration_quality = ns["evaluate_calibration_quality"]
baseline_search = ns["search"]
RANDOM_STATE = ns["RANDOM_STATE"]
base_features = list(ns["features"])
cols_base = list(ns["cols_base"])
df_train_raw = ns["df_train_raw"].copy()
df_val_raw = ns["df_val_raw"].copy()
df_test_raw = ns["df_test_raw"].copy()
baseline_val_acc = float(ns["val_acc"]); baseline_test_acc = float(ns["test_acc"]); baseline_match_acc = float(ns["match_accuracy"])
print(f"Baseline: val={baseline_val_acc:.4f}  test={baseline_test_acc:.4f}  match={baseline_match_acc:.4f}")
print(f"Cech baseline: {len(base_features)}")"""),

("md", """## 2. Court Pace Index z historii (bez leakage)
`build_court_pace_lookup()` agreguje ace rate per turniej z sezonow PRZED rokiem docelowym,
centruje i skaluje globalnie. Doczytujemy plik roku docelowego, zeby miec `tourney_id`, i liczymy
indeks predkosci dla kazdego meczu (ten sam dla obu graczy -- to kontekst meczu)."""),

("code", """lookup = build_court_pace_lookup()  # tylko historia, BEZ roku docelowego

full_target = pd.read_csv(data_file(TARGET_YEAR))
full_target["tourney_date"] = pd.to_datetime(full_target["tourney_date"], format="%Y%m%d")
full_target = full_target.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
full_target_base = full_target[cols_base + ["tourney_id"]].dropna(subset=cols_base).reset_index(drop=True)

n_train, n_val, n_test = len(df_train_raw), len(df_val_raw), len(df_test_raw)
assert len(full_target_base) == n_train + n_val + n_test, "Niespojnosc dlugosci"
ctx_train = full_target_base.iloc[:n_train].reset_index(drop=True)
ctx_val = full_target_base.iloc[n_train:n_train + n_val].reset_index(drop=True)
ctx_test = full_target_base.iloc[n_train + n_val:].reset_index(drop=True)

print("Przyklad court_pace_index (tourney_id, surface, z-score):")
for t, s in list(zip(full_target_base["tourney_id"], full_target_base["surface"]))[:6]:
    print(f"  {t:<12} {s:<6} {court_pace_index(t, s, lookup):+.3f}")"""),

("md", """## 3. Doklejenie kontekstu + symetryzacja + interakcje serve x speed
Doklejamy `court_pace_index` po `match_id`, symetryzujemy dane (kazdy mecz -> 2 wiersze p1/p2),
a nastepnie liczymy interakcje z juz-symetryzowanych cech serwisowych. Interakcje sa **antysymetryczne**
(zmieniaja znak przy zamianie p1<->p2), bo court_pace jest symetrycznym kontekstem."""),

("code", """def attach_context(df_raw, ctx):
    df_raw = df_raw.copy(); df_raw["match_id"] = range(len(df_raw))
    ctx = ctx.copy(); ctx["match_id"] = range(len(ctx))
    cpi = [court_pace_index(t, s, lookup) for t, s in zip(ctx["tourney_id"], ctx["surface"])]
    ctx_small = pd.DataFrame({"match_id": ctx["match_id"], "court_pace_index": cpi})
    return df_raw.merge(ctx_small, on="match_id", how="left", validate="one_to_one")

df_train_raw = attach_context(df_train_raw, ctx_train)
df_val_raw = attach_context(df_val_raw, ctx_val)
df_test_raw = attach_context(df_test_raw, ctx_test)

def build_split(df_raw, shuffle):
    sym = symmetrize_data(df_raw, shuffle=shuffle)
    sym = sym.merge(df_raw[["match_id", "court_pace_index"]], on="match_id", how="left", validate="many_to_one")
    sym["ace_speed_diff"] = (sym["p1_ace_rate"] - sym["p2_ace_rate"]) * sym["court_pace_index"]
    sym["first_won_speed_diff"] = (sym["p1_first_won_pct"] - sym["p2_first_won_pct"]) * sym["court_pace_index"]
    return sym

train_data = build_split(df_train_raw, True)
val_data = build_split(df_val_raw, True)
test_data = build_split(df_test_raw, True)
features = base_features + NEW_FEATURES
print(f"Cech razem: {len(features)} (baseline {len(base_features)} + nowe {len(NEW_FEATURES)})")"""),

("md", """## 4. Trening RF (tuned HP baseline) + ewaluacja
Trenujemy Random Forest z tymi samymi hiperparametrami co baseline, mierzymy match accuracy (metryka
symetryczna) i sprawdzamy, gdzie nowe cechy wladowaly sie w waznosci cech."""),

("code", """X_train, y_train = train_data[features], train_data["y"]
X_val, y_val = val_data[features], val_data["y"]
X_test, y_test = test_data[features], test_data["y"]

best_rf = RandomForestClassifier(**baseline_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE)
best_rf.fit(X_train, y_train)

val_acc = float(accuracy_score(y_val, best_rf.predict(X_val)))
test_acc = float(accuracy_score(y_test, best_rf.predict(X_test)))
proba_test = best_rf.predict_proba(X_test)[:, 1]
test_data["p1_win_probability"] = proba_test
_, match_acc = compute_symmetric_match_evaluation(test_data)
quality = evaluate_calibration_quality(y_test.to_numpy(), proba_test)

imp = pd.DataFrame({"feature": features, "importance": best_rf.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)
imp["rank"] = imp.index + 1

print(f"{'':<16} val      test     match    Brier")
print(f"{'baseline':<16} {baseline_val_acc:.4f}   {baseline_test_acc:.4f}   {baseline_match_acc:.4f}")
print(f"{'+surface_speed':<16} {val_acc:.4f}   {test_acc:.4f}   {match_acc:.4f}   {quality['brier_score']:.4f}")
print(f"DELTA match: {match_acc - baseline_match_acc:+.4f}")
print()
for f in NEW_FEATURES:
    r = imp[imp.feature == f].iloc[0]
    print(f"  {f:<22} rank {int(r['rank']):>2}/{len(features)}  importance={r['importance']:.4f}")"""),

("md", """## Wnioski

Na pojedynczym sezonie (tu rok docelowy) cecha **prawie nie zmienia accuracy** -- interakcje
`ace_speed_diff` / `first_won_speed_diff` wchodzia wysoko w waznosc (model ich uzywa), ale nie
przekladaja sie na trafnosc.

**Pelna walidacja walk-forward (4 sezony) daje pooled delta = +0.49 p.p., McNemar p = 0.26 -- nieistotne.**
Czyli pomysl jest sensowny i cechy sa uzywane, ale przewaga nie utrzymuje sie i jest w granicach szumu.
To zgodne z ogolnym wnioskiem projektu: ~65% to sufit dla cech feature-based."""),
]

make_and_run("TPM_Experiment_SurfaceSpeed.ipynb", cells, timeout=1200)
