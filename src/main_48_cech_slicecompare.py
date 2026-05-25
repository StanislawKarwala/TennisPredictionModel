"""
Compare slice-level diagnostics between the baseline model and multiple variants.

Output:
    - one XLSX workbook with the same slices for all compared models,
    - per-slice deltas showing where each variant improved or worsened vs baseline.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
import io
import os
import re
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from main_48_cech_modelslice import (
    MAX_SLICE_DEGREE,
    MIN_SUPPORT,
    compute_model_slices,
    prepare_match_level_slice_frame,
)


WORKDIR = Path(__file__).resolve().parent
BASE_DIR = WORKDIR.parent
OUTPUTS_DIR = BASE_DIR / "reports" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = OUTPUTS_DIR / "slice_comparison_all_variants.xlsx"
BASELINE_SCRIPT_PATH = (WORKDIR / "main_48_cech.py").resolve()
SLICE_COLUMNS = [
    "surface",
    "tourney_level",
    "best_of",
    "round",
    "handedness_matchup",
    "rank_gap_bucket",
    "age_gap_bucket",
    "form_gap_bucket",
]
MODELS = {
    "baseline": "main_48_cech.py",
    "sliceaware": "main_48_cech_sliceaware.py",
    "qfserve_v3": "main_48_cech_sliceaware_qfserve_v3.py",
    "bestof5_v1": "main_48_cech_sliceaware_bestof5_v1.py",
}
TARGET_SLICE_PATTERNS = [
    "best_of=5",
    "round=QF",
    "handedness_matchup=L-vs-R",
]


# Wszystkie warianty wewnetrznie wywoluja runpy.run_path na main_48_cech.py
# (przez execute_base_pipeline_quietly). Bez cache baseline pipeline -- wraz z
# RandomizedSearchCV -- byl uruchamiany 4 razy w jednym wykonaniu slicecompare.
# Po patchu uruchamia sie raz, a kolejne wywolania zwracaja ten sam slownik.
_baseline_namespace_cache: dict | None = None
_original_run_path = runpy.run_path


def _cached_run_path(path_or_name, *args, **kwargs):
    global _baseline_namespace_cache
    try:
        resolved = Path(path_or_name).resolve()
    except (TypeError, OSError):
        resolved = None

    if resolved == BASELINE_SCRIPT_PATH:
        if _baseline_namespace_cache is None:
            print("    [cache] Pierwsze uruchomienie baseline pipeline...", flush=True)
            _baseline_namespace_cache = _original_run_path(path_or_name, *args, **kwargs)
        else:
            print("    [cache] Reuse cached baseline namespace", flush=True)
        return _baseline_namespace_cache

    return _original_run_path(path_or_name, *args, **kwargs)


runpy.run_path = _cached_run_path


def execute_model_script(script_name: str) -> dict:
    script_path = WORKDIR / script_name
    original_cwd = os.getcwd()
    captured_stdout = io.StringIO()
    os.chdir(WORKDIR)
    try:
        with contextlib.redirect_stdout(captured_stdout):
            return runpy.run_path(str(script_path))
    finally:
        os.chdir(original_cwd)


def collect_slice_results(script_name: str) -> tuple[pd.DataFrame, float]:
    namespace = execute_model_script(script_name)
    match_slice_frame = prepare_match_level_slice_frame(namespace)
    slices = compute_model_slices(
        match_slice_frame,
        slice_columns=SLICE_COLUMNS,
        min_support=MIN_SUPPORT,
        max_degree=MAX_SLICE_DEGREE,
    )
    return slices, float(match_slice_frame["correct_prediction"].mean())


def build_model_view(slices_df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    return slices_df[
        [
            "slice_degree",
            "slice_definition",
            "support",
            "support_pct",
            "accuracy",
            "accuracy_gap_vs_overall",
            "avg_true_winner_probability",
        ]
    ].rename(
        columns={
            "support": f"{model_label}_support",
            "support_pct": f"{model_label}_support_pct",
            "accuracy": f"{model_label}_accuracy",
            "accuracy_gap_vs_overall": f"{model_label}_gap_vs_overall",
            "avg_true_winner_probability": f"{model_label}_avg_true_winner_probability",
        }
    )


def build_comparison_table() -> tuple[pd.DataFrame, dict[str, float]]:
    model_slices: dict[str, pd.DataFrame] = {}
    overall_metrics: dict[str, float] = {}
    total_models = len(MODELS)
    for index, (model_label, script_name) in enumerate(MODELS.items(), start=1):
        print(
            f"[{index}/{total_models}] Licze model: {model_label} ({script_name})...",
            flush=True,
        )
        slices, overall = collect_slice_results(script_name)
        model_slices[model_label] = slices
        overall_metrics[model_label] = overall
        print(
            f"    gotowe: match_accuracy={overall:.4f}, liczba_slice={len(slices)}",
            flush=True,
        )

    baseline_view = build_model_view(model_slices["baseline"], "baseline")
    comparison = baseline_view.copy()

    for model_label in MODELS:
        if model_label == "baseline":
            continue

        model_view = build_model_view(model_slices[model_label], model_label)
        comparison = comparison.merge(
            model_view,
            on=["slice_degree", "slice_definition"],
            how="inner",
            validate="one_to_one",
        )

        if len(comparison) != len(baseline_view) or len(comparison) != len(model_view):
            raise ValueError(f"Slice sets differ between baseline and {model_label}.")

        if not np.allclose(comparison["baseline_support"], comparison[f"{model_label}_support"]):
            raise ValueError(f"Support mismatch between baseline and {model_label} slices.")

        comparison[f"{model_label}_accuracy_delta_vs_baseline"] = (
            comparison[f"{model_label}_accuracy"] - comparison["baseline_accuracy"]
        )
        comparison[f"{model_label}_gap_delta_vs_baseline"] = (
            comparison[f"{model_label}_gap_vs_overall"] - comparison["baseline_gap_vs_overall"]
        )
        comparison[f"{model_label}_probability_delta_vs_baseline"] = (
            comparison[f"{model_label}_avg_true_winner_probability"]
            - comparison["baseline_avg_true_winner_probability"]
        )

    comparison["support"] = comparison["baseline_support"].astype(int)
    comparison["support_pct"] = comparison["baseline_support_pct"]

    target_regex = "|".join(re.escape(pattern) for pattern in TARGET_SLICE_PATTERNS)
    comparison["is_target_slice"] = comparison["slice_definition"].str.contains(
        target_regex,
        regex=True,
    )

    accuracy_columns = [f"{model_label}_accuracy" for model_label in MODELS]
    comparison["best_model_by_accuracy"] = (
        comparison[accuracy_columns]
        .idxmax(axis=1)
        .str.replace("_accuracy", "", regex=False)
    )
    comparison["best_accuracy"] = comparison[accuracy_columns].max(axis=1)

    if "qfserve_v3_accuracy_delta_vs_baseline" in comparison.columns:
        primary_sort_column = "qfserve_v3_accuracy_delta_vs_baseline"
    else:
        primary_sort_column = next(
            column for column in comparison.columns if column.endswith("_accuracy_delta_vs_baseline")
        )

    ordered_columns = [
        "slice_degree",
        "slice_definition",
        "is_target_slice",
        "support",
        "support_pct",
        "best_model_by_accuracy",
        "best_accuracy",
        "baseline_accuracy",
        "baseline_gap_vs_overall",
        "baseline_avg_true_winner_probability",
    ]

    for model_label in MODELS:
        if model_label == "baseline":
            continue
        ordered_columns.extend(
            [
                f"{model_label}_accuracy",
                f"{model_label}_accuracy_delta_vs_baseline",
                f"{model_label}_gap_vs_overall",
                f"{model_label}_gap_delta_vs_baseline",
                f"{model_label}_avg_true_winner_probability",
                f"{model_label}_probability_delta_vs_baseline",
            ]
        )

    comparison = comparison[ordered_columns].sort_values(
        ["is_target_slice", primary_sort_column, "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    rounded_columns = [column for column in comparison.columns if column not in {
        "slice_degree",
        "slice_definition",
        "is_target_slice",
        "support",
        "best_model_by_accuracy",
    }]
    comparison[rounded_columns] = comparison[rounded_columns].round(4)
    return comparison, overall_metrics


def build_overall_metrics_frame(overall_metrics: dict[str, float]) -> pd.DataFrame:
    baseline_value = overall_metrics["baseline"]
    return pd.DataFrame(
        [
            {
                "model": model_label,
                "match_accuracy": value,
                "delta_vs_baseline": value - baseline_value,
            }
            for model_label, value in overall_metrics.items()
        ]
    ).round(4)


def autosize_worksheet(worksheet) -> None:
    for column_cells in worksheet.columns:
        values = [cell.value for cell in column_cells if cell.value is not None]
        if not values:
            continue
        column_letter = column_cells[0].column_letter
        max_length = max(len(str(value)) for value in values)
        worksheet.column_dimensions[column_letter].width = min(max_length + 2, 40)


def build_fallback_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_XLSX.with_name(f"{OUTPUT_XLSX.stem}_{timestamp}{OUTPUT_XLSX.suffix}")


def save_comparison_workbook(
    comparison: pd.DataFrame,
    overall_metrics: dict[str, float],
) -> Path:
    overall_df = build_overall_metrics_frame(overall_metrics)

    output_path = OUTPUT_XLSX
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            overall_df.to_excel(writer, sheet_name="overall_metrics", index=False)
            comparison.to_excel(writer, sheet_name="slice_comparison", index=False)

            for worksheet in writer.sheets.values():
                worksheet.freeze_panes = "A2"
                worksheet.auto_filter.ref = worksheet.dimensions
                autosize_worksheet(worksheet)
    except PermissionError:
        output_path = build_fallback_output_path()
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            overall_df.to_excel(writer, sheet_name="overall_metrics", index=False)
            comparison.to_excel(writer, sheet_name="slice_comparison", index=False)

            for worksheet in writer.sheets.values():
                worksheet.freeze_panes = "A2"
                worksheet.auto_filter.ref = worksheet.dimensions
                autosize_worksheet(worksheet)

    return output_path


def print_target_slice_summary(
    comparison: pd.DataFrame,
    overall_metrics: dict[str, float],
    output_path: Path,
) -> None:
    print("#" * 70)
    print("SLICE COMPARISON: BASELINE VS VARIANTS")
    print("#" * 70)
    baseline_value = overall_metrics["baseline"]
    summary_parts = [f"baseline={baseline_value:.4f}"]
    for model_label, value in overall_metrics.items():
        if model_label == "baseline":
            continue
        summary_parts.append(
            f"{model_label}={value:.4f} ({value - baseline_value:+.4f})"
        )
    print("Overall match accuracy: " + " | ".join(summary_parts))
    print(f"Excel output: {output_path.name}")
    print()

    target_rows = comparison[comparison["is_target_slice"]].copy()
    target_columns = ["slice_definition", "support", "baseline_accuracy"]
    for model_label in MODELS:
        if model_label == "baseline":
            continue
        target_columns.extend(
            [
                f"{model_label}_accuracy",
                f"{model_label}_accuracy_delta_vs_baseline",
            ]
        )
    target_columns.append("best_model_by_accuracy")

    print("Najwazniejsze target slice'y:")
    print(
        target_rows[target_columns].to_string(
            index=False,
            formatters={
                "baseline_accuracy": lambda value: f"{100 * value:5.1f}%",
                **{
                    column: (lambda value: f"{100 * value:5.1f}%")
                    for column in target_columns
                    if column.endswith("_accuracy")
                    and column not in {"baseline_accuracy", "best_model_by_accuracy"}
                },
                **{
                    column: (lambda value: f"{100 * value:+5.1f} p.p.")
                    for column in target_columns
                    if column.endswith("_accuracy_delta_vs_baseline")
                },
            },
        )
    )
    print()

    for model_label in MODELS:
        if model_label == "baseline":
            continue

        gain_column = f"{model_label}_accuracy_delta_vs_baseline"
        biggest_gains = comparison.sort_values(gain_column, ascending=False).head(8)
        biggest_losses = comparison.sort_values(gain_column, ascending=True).head(8)

        print(f"Najwieksze zyski accuracy dla {model_label}:")
        print(
            biggest_gains[
                ["slice_definition", "support", gain_column]
            ].to_string(
                index=False,
                formatters={
                    gain_column: lambda value: f"{100 * value:+5.1f} p.p.",
                },
            )
        )
        print()

        print(f"Najwieksze spadki accuracy dla {model_label}:")
        print(
            biggest_losses[
                ["slice_definition", "support", gain_column]
            ].to_string(
                index=False,
                formatters={
                    gain_column: lambda value: f"{100 * value:+5.1f} p.p.",
                },
            )
        )
        print()


def main() -> None:
    comparison, overall_metrics = build_comparison_table()
    output_path = save_comparison_workbook(comparison, overall_metrics)
    print_target_slice_summary(comparison, overall_metrics, output_path)


if __name__ == "__main__":
    main()