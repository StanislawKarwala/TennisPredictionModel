"""
Seed stability dla tennis_model.py.

Uruchamia baseline pipeline N razy z roznymi RANDOM_STATE i agreguje wyniki:
val_acc, test_acc, match_accuracy oraz (jezeli dostepne) match_accuracy_tuned
po kalibracji.

Cel: zobaczyc rozrzut accuracy z powodu losowego ziarna. Jezeli baseline
ma std=1.5 p.p. a sliceaware-vs-baseline daje delta 0.3 p.p., to delta
jest w szumie i nie ma sensu interpretowac jej jakosciowo.

Uzycie:
    python tennis_model_seedstability.py
    # albo z customowymi seedami:
    TENNIS_SEEDS="1,7,13,42,100" python tennis_model_seedstability.py
"""

from __future__ import annotations

import contextlib
import io
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


BASE_SCRIPT = Path(__file__).with_name("tennis_model.py")
DEFAULT_SEEDS = [13, 42, 99, 100, 123, 256, 777, 1024, 2024, 31337]


def parse_seeds() -> list[int]:
    raw = os.environ.get("TENNIS_SEEDS")
    if not raw:
        return DEFAULT_SEEDS
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def run_one_seed(seed: int) -> dict[str, float]:
    """Uruchamia baseline pipeline z podanym seedem i zbiera kluczowe metryki."""
    original_cwd = os.getcwd()
    original_seed = os.environ.get("TENNIS_RANDOM_STATE")
    captured_stdout = io.StringIO()

    os.environ["TENNIS_RANDOM_STATE"] = str(seed)
    os.chdir(BASE_SCRIPT.parent)
    try:
        with contextlib.redirect_stdout(captured_stdout):
            namespace = runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)
        if original_seed is None:
            os.environ.pop("TENNIS_RANDOM_STATE", None)
        else:
            os.environ["TENNIS_RANDOM_STATE"] = original_seed

    # cv_accuracy jest teraz eksportowane osobno przez tennis_model.py, bo
    # search.best_score_ to neg_log_loss (refit po log-loss, nie accuracy).
    cv_accuracy = namespace.get("cv_accuracy")
    if cv_accuracy is None:
        # Fallback dla starszych wersji baseline. UWAGA: best_score_ wolno uzyc
        # tylko gdy scoring to accuracy -- przy refit='neg_log_loss' best_score_
        # jest ujemnym log-lossem i wstawienie go do kolumny cv_accuracy
        # falszowaloby podsumowanie (mean/std liczone z dwoch roznych metryk).
        search = namespace["search"]
        cv_results = getattr(search, "cv_results_", {})
        if "mean_test_accuracy" in cv_results:
            cv_accuracy = float(cv_results["mean_test_accuracy"][search.best_index_])
        elif getattr(search, "scoring", None) == "accuracy":
            cv_accuracy = float(search.best_score_)
        else:
            cv_accuracy = float("nan")
    result = {
        "seed": seed,
        "cv_accuracy": float(cv_accuracy),
        "val_accuracy": float(namespace["val_acc"]),
        "test_accuracy": float(namespace["test_acc"]),
        "match_accuracy": float(namespace["match_accuracy"]),
    }
    if "match_accuracy_tuned" in namespace:
        result["match_accuracy_tuned"] = float(namespace["match_accuracy_tuned"])
    if "best_threshold" in namespace:
        result["best_threshold"] = float(namespace["best_threshold"])
    return result


def summarize(results: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    metric_columns = [col for col in df.columns if col != "seed"]
    summary = pd.DataFrame(
        {
            "metric": metric_columns,
            "mean": [df[col].mean() for col in metric_columns],
            "std": [df[col].std(ddof=1) for col in metric_columns],
            "min": [df[col].min() for col in metric_columns],
            "max": [df[col].max() for col in metric_columns],
            "range_pp": [(df[col].max() - df[col].min()) * 100 for col in metric_columns],
        }
    )
    return summary


def main() -> None:
    seeds = parse_seeds()
    print(f"Uruchamiam baseline pipeline dla {len(seeds)} seedow: {seeds}")
    print("(kazdy run = full RandomizedSearchCV + final fit + ewaluacja)")
    print()

    results: list[dict[str, float]] = []
    for index, seed in enumerate(seeds, start=1):
        print(f"[{index}/{len(seeds)}] Seed = {seed}...", flush=True)
        try:
            metrics = run_one_seed(seed)
        except Exception as exc:
            print(f"    BLAD: {exc}")
            continue
        results.append(metrics)
        print(
            f"    cv={metrics['cv_accuracy']:.4f} "
            f"| val={metrics['val_accuracy']:.4f} "
            f"| test={metrics['test_accuracy']:.4f} "
            f"| match={metrics['match_accuracy']:.4f}"
            + (
                f" | tuned={metrics['match_accuracy_tuned']:.4f}"
                if "match_accuracy_tuned" in metrics
                else ""
            )
        )

    if not results:
        print("Brak wynikow do podsumowania.")
        return

    per_seed = pd.DataFrame(results)
    summary = summarize(results)

    print()
    print("=" * 70)
    print("WYNIKI PER-SEED")
    print("=" * 70)
    print(per_seed.to_string(index=False))

    print()
    print("=" * 70)
    print("PODSUMOWANIE STABILNOSCI")
    print("=" * 70)
    print(summary.to_string(index=False))

    outputs_dir = Path(__file__).resolve().parents[1] / "reports" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "seed_stability_results.csv"
    per_seed.to_csv(output_path, index=False)
    summary_path = outputs_dir / "seed_stability_summary.csv"
    summary.to_csv(summary_path, index=False)
    print()
    print(f"Per-seed zapisane: {output_path.name}")
    print(f"Summary zapisane:  {summary_path.name}")
    print()
    print("Interpretacja: jezeli std match_accuracy > delta sliceaware-vs-baseline,")
    print("to delta jest w szumie i nie ma istotnosci.")


if __name__ == "__main__":
    main()
