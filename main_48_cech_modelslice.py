"""
Model slicing dla modelu z main_48_cech.py
=========================================

Cel:
  - uruchomic istniejący pipeline treningowo-ewaluacyjny bez zmian,
  - policzyc jak model dziala na semantycznych podgrupach danych testowych,
  - znalezc slice'y, na ktorych model wyraznie niedomaga.

Inspiracja:
  - GuideAI25_2.pdf ("Model Slicing for Responsible AI")
  - https://github.com/yeounoh/slicefinder
  - https://github.com/microsoft/responsible-ai-toolbox

To nie jest wierna implementacja SliceFinder. To pragmatyczna warstwa
diagnostyczna dla modelu tenisowego: liczymy accuracy dla podgrup meczow
oraz porownujemy je z accuracy ogolnym na poziomie meczu.
"""

from __future__ import annotations

import itertools
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


BASE_SCRIPT = Path(__file__).with_name(
    os.environ.get("TENNIS_MODEL_SLICE_BASE", "main_48_cech.py")
)

# Minimalny support chroni przed nadinterpretacja bardzo malych slice'ow.
MIN_SUPPORT = 5
MAX_SLICE_DEGREE = 2
UNDERPERFORMANCE_GAP = -0.05
TOP_N = 12


def execute_base_pipeline() -> dict:
    """Run the existing model script and return its namespace."""
    original_cwd = os.getcwd()
    os.chdir(BASE_SCRIPT.parent)
    try:
        return runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)


def require_namespace_items(namespace: dict, names: list[str]) -> None:
    missing = [name for name in names if name not in namespace]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(
            "Brakuje wymaganych obiektow po uruchomieniu main_48_cech.py: "
            f"{joined}"
        )


def build_handedness_matchup(row: pd.Series) -> str:
    hands = sorted([row["winner_hand"], row["loser_hand"]])
    return f"{hands[0]}-vs-{hands[1]}"


def build_bucketed_feature(
    series: pd.Series,
    bins: list[float],
    labels: list[str],
) -> pd.Series:
    return pd.cut(
        series,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype("object")


def prepare_match_level_slice_frame(namespace: dict) -> pd.DataFrame:
    """
    Build one-row-per-match evaluation table for slicing.

    We slice on real matches, not on symmetrized rows, because responsible
    debugging should inspect the model on actual decision cases.
    """
    require_namespace_items(
        namespace,
        ["df_test_raw", "winner_perspective", "match_accuracy"],
    )

    df_test_raw = namespace["df_test_raw"].copy()
    winner_perspective = namespace["winner_perspective"].copy()
    reported_match_accuracy = float(namespace["match_accuracy"])

    match_context = df_test_raw[
        [
            "match_id",
            "surface",
            "tourney_level",
            "best_of",
            "round",
            "winner_hand",
            "loser_hand",
            "winner_rank",
            "loser_rank",
            "winner_age",
            "loser_age",
            "w_form",
            "l_form",
        ]
    ].copy()

    match_context["handedness_matchup"] = match_context.apply(
        build_handedness_matchup,
        axis=1,
    )
    match_context["rank_gap_bucket"] = build_bucketed_feature(
        (match_context["winner_rank"] - match_context["loser_rank"]).abs(),
        bins=[-0.1, 10, 25, 50, 100, np.inf],
        labels=["0-10", "11-25", "26-50", "51-100", ">100"],
    )
    match_context["age_gap_bucket"] = build_bucketed_feature(
        (match_context["winner_age"] - match_context["loser_age"]).abs(),
        bins=[-0.1, 2, 5, 8, np.inf],
        labels=["0-2", "3-5", "6-8", ">8"],
    )
    match_context["form_gap_bucket"] = build_bucketed_feature(
        (match_context["w_form"] - match_context["l_form"]).abs(),
        bins=[-0.001, 0.10, 0.25, 0.40, 1.0],
        labels=["0.00-0.10", "0.10-0.25", "0.25-0.40", ">0.40"],
    )

    evaluation = winner_perspective[
        [
            "match_id",
            "correct_prediction",
            "p1_win_probability",
            "predicted_winner",
            "actual_winner",
        ]
    ].copy()
    evaluation["correct_prediction"] = evaluation["correct_prediction"].astype(int)

    slice_frame = match_context.merge(
        evaluation,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    computed_accuracy = float(slice_frame["correct_prediction"].mean())
    if not np.isclose(computed_accuracy, reported_match_accuracy):
        raise ValueError(
            "Niespojnosc: accuracy match-level po joinie rozni sie od match_accuracy "
            f"({computed_accuracy:.6f} vs {reported_match_accuracy:.6f})."
        )

    return slice_frame


def slice_description(columns: tuple[str, ...], values: tuple[object, ...]) -> str:
    return " & ".join(f"{column}={value}" for column, value in zip(columns, values))


def compute_model_slices(
    match_slice_frame: pd.DataFrame,
    slice_columns: list[str],
    min_support: int = MIN_SUPPORT,
    max_degree: int = MAX_SLICE_DEGREE,
) -> pd.DataFrame:
    overall_accuracy = float(match_slice_frame["correct_prediction"].mean())
    total_matches = len(match_slice_frame)

    rows: list[dict] = []
    for degree in range(1, max_degree + 1):
        for combo in itertools.combinations(slice_columns, degree):
            grouped = match_slice_frame.groupby(list(combo), dropna=False, observed=True)
            for group_key, group_df in grouped:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)

                support = len(group_df)
                if support < min_support:
                    continue

                accuracy = float(group_df["correct_prediction"].mean())
                error_rate = 1.0 - accuracy
                avg_true_winner_probability = float(group_df["p1_win_probability"].mean())

                row = {
                    "slice_degree": degree,
                    "slice_definition": slice_description(combo, group_key),
                    "support": support,
                    "support_pct": support / total_matches,
                    "accuracy": accuracy,
                    "error_rate": error_rate,
                    "accuracy_gap_vs_overall": accuracy - overall_accuracy,
                    "avg_true_winner_probability": avg_true_winner_probability,
                }
                for column, value in zip(combo, group_key):
                    row[column] = value
                rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "slice_degree",
                "slice_definition",
                "support",
                "support_pct",
                "accuracy",
                "error_rate",
                "accuracy_gap_vs_overall",
                "avg_true_winner_probability",
            ]
        )

    results = pd.DataFrame(rows)
    return results.sort_values(
        ["accuracy", "support"],
        ascending=[True, False],
    ).reset_index(drop=True)


def print_slice_table(
    title: str,
    slices_df: pd.DataFrame,
    top_n: int = TOP_N,
    gap_threshold: float = UNDERPERFORMANCE_GAP,
) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    problematic = slices_df[slices_df["accuracy_gap_vs_overall"] <= gap_threshold].copy()
    if problematic.empty:
        print(
            "Brak slice'ow spelniajacych prog underperformance. "
            "Pokazuje najslabsze slice'y wedlug accuracy."
        )
        problematic = slices_df.head(top_n).copy()
    else:
        problematic = problematic.head(top_n).copy()

    display_columns = [
        "slice_definition",
        "support",
        "support_pct",
        "accuracy",
        "accuracy_gap_vs_overall",
        "avg_true_winner_probability",
    ]
    print(
        problematic[display_columns].to_string(
            index=False,
            formatters={
                "support_pct": lambda value: f"{100 * value:5.1f}%",
                "accuracy": lambda value: f"{100 * value:5.1f}%",
                "accuracy_gap_vs_overall": lambda value: f"{100 * value:+5.1f} p.p.",
                "avg_true_winner_probability": lambda value: f"{value:.3f}",
            },
        )
    )


def print_best_slices(slices_df: pd.DataFrame, top_n: int = 8) -> None:
    print("\n" + "=" * 70)
    print("NAJLEPSZE SLICE (dla kontrastu)")
    print("=" * 70)
    best = slices_df.sort_values(["accuracy", "support"], ascending=[False, False]).head(top_n)
    print(
        best[
            [
                "slice_definition",
                "support",
                "support_pct",
                "accuracy",
                "accuracy_gap_vs_overall",
            ]
        ].to_string(
            index=False,
            formatters={
                "support_pct": lambda value: f"{100 * value:5.1f}%",
                "accuracy": lambda value: f"{100 * value:5.1f}%",
                "accuracy_gap_vs_overall": lambda value: f"{100 * value:+5.1f} p.p.",
            },
        )
    )


def run_model_slicing() -> None:
    namespace = execute_base_pipeline()
    match_slice_frame = prepare_match_level_slice_frame(namespace)

    slice_columns = [
        "surface",
        "tourney_level",
        "best_of",
        "round",
        "handedness_matchup",
        "rank_gap_bucket",
        "age_gap_bucket",
        "form_gap_bucket",
    ]

    slice_results = compute_model_slices(
        match_slice_frame,
        slice_columns=slice_columns,
        min_support=MIN_SUPPORT,
        max_degree=MAX_SLICE_DEGREE,
    )

    overall_accuracy = float(match_slice_frame["correct_prediction"].mean())
    print("\n" + "#" * 70)
    print("MODEL SLICING FOR RESPONSIBLE AI")
    print("#" * 70)
    print(f"Analizowany skrypt: {BASE_SCRIPT.name}")
    print(
        "Model slicing = sprawdzenie, dla jakich podgrup meczow model dziala "
        "gorzej lub lepiej niz srednio."
    )
    print(f"Liczba realnych meczow w analizie: {len(match_slice_frame)}")
    print(f"Accuracy ogolne (match-level):     {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    print(f"Minimalny support slice'a:         {MIN_SUPPORT}")
    print(f"Maksymalny stopien slice'a:        {MAX_SLICE_DEGREE}D")
    print(
        "Analizowane atrybuty slice'ow: "
        + ", ".join(slice_columns)
    )

    one_dimensional = slice_results[slice_results["slice_degree"] == 1].reset_index(drop=True)
    two_dimensional = slice_results[slice_results["slice_degree"] == 2].reset_index(drop=True)

    print_slice_table("NAJSLABSZE SLICE 1D", one_dimensional)
    print_slice_table("NAJSLABSZE SLICE 2D", two_dimensional)
    print_best_slices(slice_results)


if __name__ == "__main__":
    run_model_slicing()