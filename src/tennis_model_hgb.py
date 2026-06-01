"""
Eksperyment Sprint 2: HistGradientBoosting vs Random Forest
===========================================================

Cel:
  Sprawdzic czy gradient boosting (HistGradientBoostingClassifier z sklearn --
  zero nowych zaleznosci) bije Random Forest na DOKLADNIE tych samych danych
  i cechach co baseline. To uczciwe porownanie (ablation): jedyna zmiana to
  algorytm, wszystko inne (cechy, split, symetryzacja, metryka) bez zmian.

Metodologia:
  - reuzywamy pipeline z tennis_model.py (przez runpy) -> te same df_train_raw,
    val_data, test_data, features, oraz pomocnicze compute_symmetric_match_evaluation
    i evaluate_calibration_quality (metryka symetryczna z Sprint 1),
  - dobor hiperparametrow HGB przez RandomizedSearchCV ze scoringiem neg_log_loss
    na TimeSeriesSplit (chronologicznie), spojnie z baseline po Sprint 1 (C2),
  - finalny model trenowany na pelnym, wymieszanym zbiorze treningowym,
  - porownanie val/test/match accuracy + Brier/log-loss/ECE: HGB vs RF.

Jezeli HGB wygrywa wyraznie i poza szumem -- warto uczynic go domyslnym modelem
baseline'u (i wariantow). Jezeli nie -- zostajemy przy RF.
"""

from __future__ import annotations

import contextlib
import io
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score


BASE_SCRIPT = Path(__file__).with_name("tennis_model.py")


def execute_base_pipeline_quietly() -> dict:
    """Uruchamia baseline z wyciszonym stdout i zwraca jego namespace."""
    original_cwd = os.getcwd()
    captured = io.StringIO()
    os.chdir(BASE_SCRIPT.parent)
    try:
        with contextlib.redirect_stdout(captured):
            return runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)


def print_row(label: str, val_acc: float, test_acc: float, match_acc: float,
              brier: float, logloss: float, ece: float) -> None:
    print(
        f"{label:<16} val={val_acc:.4f}  test={test_acc:.4f}  match={match_acc:.4f}  "
        f"Brier={brier:.4f}  logloss={logloss:.4f}  ECE={ece:.4f}"
    )


def tune_and_eval_hgb(
    *,
    label,
    features,
    X_train_cv,
    y_train_cv,
    df_train_raw,
    symmetrize_data,
    X_val,
    y_val,
    X_test,
    y_test,
    test_data,
    compute_symmetric_match_evaluation,
    evaluate_calibration_quality,
    RANDOM_STATE,
    categorical_features=None,
):
    """Stroi i ocenia jeden wariant HGB. categorical_features = lista nazw kolumn
    traktowanych natywnie jako kategorie (HGB nie zaklada wtedy porzadku liczb,
    co naprawia problem LabelEncoder dla surface/tourney_level)."""
    print(f"\nStrojenie {label} (RandomizedSearchCV, neg_log_loss)...")
    hgb = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        early_stopping=False,
        categorical_features=categorical_features,
    )
    hgb_param_dist = {
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_iter": [200, 300, 400, 600],
        "max_leaf_nodes": [15, 31, 63],
        "max_depth": [None, 3, 5, 8],
        "min_samples_leaf": [20, 40, 80, 120],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_features": [0.6, 0.8, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        hgb,
        hgb_param_dist,
        n_iter=40,
        cv=tscv,
        scoring={"neg_log_loss": "neg_log_loss", "accuracy": "accuracy", "roc_auc": "roc_auc"},
        refit="neg_log_loss",
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train_cv, y_train_cv)
    cv_nll = float(search.best_score_)
    cv_acc = float(search.cv_results_["mean_test_accuracy"][search.best_index_])
    cv_auc = float(search.cv_results_["mean_test_roc_auc"][search.best_index_])
    print(f"  najlepsze HP: {search.best_params_}")
    print(f"  CV: neg_log_loss={cv_nll:.4f} | accuracy={cv_acc:.4f} | roc_auc={cv_auc:.4f}")

    train_final = symmetrize_data(df_train_raw, shuffle=True)
    best = search.best_estimator_
    best.fit(train_final[features], train_final["y"])

    val_acc = float(accuracy_score(y_val, best.predict(X_val)))
    test_acc = float(accuracy_score(y_test, best.predict(X_test)))
    proba_test = best.predict_proba(X_test)[:, 1]
    td = test_data.copy()
    td["p1_win_probability"] = proba_test
    _, match_acc = compute_symmetric_match_evaluation(td)
    quality = evaluate_calibration_quality(y_test.to_numpy(), proba_test)
    return {
        "label": label,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "match_acc": match_acc,
        "brier": quality["brier_score"],
        "logloss": quality["log_loss"],
        "ece": quality["expected_calibration_error"],
    }


def main() -> None:
    print("Uruchamiam baseline pipeline (reuzycie danych i cech)...")
    ns = execute_base_pipeline_quietly()

    features = ns["features"]
    symmetrize_data = ns["symmetrize_data"]
    compute_symmetric_match_evaluation = ns["compute_symmetric_match_evaluation"]
    evaluate_calibration_quality = ns["evaluate_calibration_quality"]
    RANDOM_STATE = ns["RANDOM_STATE"]

    df_train_raw = ns["df_train_raw"]
    test_data = ns["test_data"]
    X_val, y_val = ns["X_val"], ns["y_val"]
    X_test, y_test = ns["X_test"], ns["y_test"]
    X_train_cv, y_train_cv = ns["X_train_cv"], ns["y_train_cv"]

    # --- Baseline RF (juz policzony w namespace) ---
    best_rf = ns["best_rf"]
    rf_val_acc = float(ns["val_acc"])
    rf_test_acc = float(ns["test_acc"])
    rf_match_acc = float(ns["match_accuracy"])
    rf_proba_test = best_rf.predict_proba(X_test)[:, 1]
    rf_quality = evaluate_calibration_quality(y_test.to_numpy(), rf_proba_test)

    # =========================================================================
    # HistGradientBoosting -- dwa warianty
    # =========================================================================
    # early_stopping=False: robimy wlasne CV (TimeSeriesSplit), nie chcemy
    # zeby HGB wycinal sobie wewnetrzny zbior walidacyjny i mieszal chronologie.
    common = dict(
        features=features,
        X_train_cv=X_train_cv,
        y_train_cv=y_train_cv,
        df_train_raw=df_train_raw,
        symmetrize_data=symmetrize_data,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_data=test_data,
        compute_symmetric_match_evaluation=compute_symmetric_match_evaluation,
        evaluate_calibration_quality=evaluate_calibration_quality,
        RANDOM_STATE=RANDOM_STATE,
    )

    # Wariant 1: cechy jak RF (surface/tourney_level jako liczby z LabelEncoder).
    hgb_num = tune_and_eval_hgb(label="HGB (numeric)", categorical_features=None, **common)

    # Wariant 2: NATYWNE kategorie -- surface i tourney_level traktowane jako
    # nominalne (HGB nie zaklada porzadku alfabetycznego LabelEncoder). To glowna
    # przewaga HGB nad RF, ktorej wariant 1 nie wykorzystywal.
    cat_cols = [c for c in ("surface", "tourney_level") if c in features]
    hgb_cat = tune_and_eval_hgb(
        label="HGB (kategorie)", categorical_features=cat_cols, **common
    )

    # =========================================================================
    # PODSUMOWANIE
    # =========================================================================
    print("\n" + "=" * 82)
    print("POROWNANIE: Random Forest (baseline) vs HistGradientBoosting")
    print("=" * 82)
    print_row(
        "RandomForest", rf_val_acc, rf_test_acc, rf_match_acc,
        rf_quality["brier_score"], rf_quality["log_loss"],
        rf_quality["expected_calibration_error"],
    )
    for r in (hgb_num, hgb_cat):
        print_row(
            r["label"], r["val_acc"], r["test_acc"], r["match_acc"],
            r["brier"], r["logloss"], r["ece"],
        )
    print("-" * 82)
    for r in (hgb_num, hgb_cat):
        print(
            f"DELTA ({r['label']} - RF): val={r['val_acc'] - rf_val_acc:+.4f}  "
            f"test={r['test_acc'] - rf_test_acc:+.4f}  "
            f"match={r['match_acc'] - rf_match_acc:+.4f}  "
            f"Brier={r['brier'] - rf_quality['brier_score']:+.4f}"
        )
    print()
    print(
        "UWAGA: przy ~590 meczach testowych CI dla match accuracy ~ +/-4 p.p. "
        "Roznice ponizej tego progu sa w szumie -- ostateczna ocena dopiero w "
        "Sprint 4 (walk-forward na wielu latach)."
    )


if __name__ == "__main__":
    main()
