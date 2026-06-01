"""
Slice-aware extension of tennis_model.py.

Idea:
  - reuse the same base preprocessing and split as the baseline model,
  - add context features for the weakest slices found by model slicing,
  - train a second Random Forest with the same tuned hyperparameters,
  - compare the new model against the baseline on the same validation/test split.

Targeted weak slices:
  - Best of 5
  - QF
  - L-vs-R
"""

from __future__ import annotations

import bisect
import contextlib
import io
import os
import runpy
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_SCRIPT = Path(__file__).with_name("tennis_model.py")
LATE_ROUNDS = {"QF", "SF", "BR", "F"}

TARGETED_FEATURES = [
    "is_best_of5",
    "is_qf",
    "is_lefty_matchup",
    "p1_best_of5_form",
    "p2_best_of5_form",
    "best_of5_form_diff",
    "p1_best_of5_experience",
    "p2_best_of5_experience",
    "best_of5_experience_diff",
    "p1_late_round_form",
    "p2_late_round_form",
    "late_round_form_diff",
    "p1_late_round_experience",
    "p2_late_round_experience",
    "late_round_experience_diff",
    "p1_vs_opp_hand_form",
    "p2_vs_opp_hand_form",
    "opp_hand_form_diff",
    "p1_qf_form",
    "p2_qf_form",
    "qf_form_diff",
    "p1_qf_experience",
    "p2_qf_experience",
    "qf_experience_diff",
    "p1_qf_surface_form",
    "p2_qf_surface_form",
    "qf_surface_form_diff",
    "p1_vs_opp_hand_surface_form",
    "p2_vs_opp_hand_surface_form",
    "opp_hand_surface_form_diff",
    "p1_vs_opp_hand_balance",
    "p2_vs_opp_hand_balance",
    "opp_hand_balance_diff",
]


def execute_base_pipeline_quietly() -> dict:
    """Run the baseline script without printing its full console output."""
    original_cwd = os.getcwd()
    captured_stdout = io.StringIO()
    os.chdir(BASE_SCRIPT.parent)
    try:
        with contextlib.redirect_stdout(captured_stdout):
            return runpy.run_path(str(BASE_SCRIPT))
    finally:
        os.chdir(original_cwd)


def build_player_index(full_sequence: pd.DataFrame) -> dict[str, list[int]]:
    """
    player_name -> posortowana lista absolutnych indeksow wierszy gracza
    w full_sequence. Pozwala dla cutoff i robic bisect_left zamiast skanowac
    cala historie. Per-mecz zysk: ~19k -> ~100-200 wierszy do filtrowania
    przez best_of/surface/opponent_hand.
    """
    winners = full_sequence["winner_name"].to_numpy()
    losers = full_sequence["loser_name"].to_numpy()
    indices: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(full_sequence)):
        indices[winners[idx]].append(idx)
        if losers[idx] != winners[idx]:
            indices[losers[idx]].append(idx)
    return indices


def get_player_history_via_index(
    player_name: str,
    full_sequence: pd.DataFrame,
    player_index: dict[str, list[int]],
    cutoff: int,
) -> pd.DataFrame:
    """
    Zwraca slice full_sequence dla wszystkich meczow gracza o absolutnym
    indeksie < cutoff. Bisect daje pozycje pierwszego indeksu >= cutoff.
    """
    all_indices = player_index.get(player_name, [])
    if not all_indices:
        return full_sequence.iloc[[]]
    end = bisect.bisect_left(all_indices, cutoff)
    if end == 0:
        return full_sequence.iloc[[]]
    return full_sequence.iloc[all_indices[:end]]


def get_player_history(player_name: str, history: pd.DataFrame) -> pd.DataFrame:
    """Kept for backward compatibility -- legacy callers without precomputed index."""
    return history[
        (history["winner_name"] == player_name) |
        (history["loser_name"] == player_name)
    ]


def _apply_context_filters(
    player_history: pd.DataFrame,
    player_name: str,
    *,
    best_of: int | None,
    rounds: set[str] | None,
    surface: str | None,
    opponent_hand: str | None,
) -> pd.DataFrame:
    """
    Stosuje filtry kontekstowe na juz-przefiltrowanym subsecie gracza.
    Wyniesione zeby uniknac duplikacji miedzy form/experience/balance.
    """
    if best_of is not None:
        player_history = player_history[player_history["best_of"] == best_of]
    if rounds is not None:
        player_history = player_history[player_history["round"].isin(rounds)]
    if surface is not None:
        player_history = player_history[player_history["surface"] == surface]
    if opponent_hand is not None:
        versus_hand_mask = (
            ((player_history["winner_name"] == player_name) & (player_history["loser_hand"] == opponent_hand)) |
            ((player_history["loser_name"] == player_name) & (player_history["winner_hand"] == opponent_hand))
        )
        player_history = player_history[versus_hand_mask]
    return player_history


def calculate_context_form(
    player_name: str,
    history: pd.DataFrame,
    *,
    best_of: int | None = None,
    rounds: set[str] | None = None,
    surface: str | None = None,
    opponent_hand: str | None = None,
    window: int = 12,
    min_matches: int = 3,
    fallback: float = 0.5,
) -> float:
    """
    `history` to juz przefiltrowany subset gracza (przez get_player_history_via_index).
    Pozostale filtry stosowane sa lokalnie.
    """
    player_history = _apply_context_filters(
        history,
        player_name,
        best_of=best_of,
        rounds=rounds,
        surface=surface,
        opponent_hand=opponent_hand,
    ).tail(window)
    if len(player_history) < min_matches:
        return fallback

    wins = (player_history["winner_name"] == player_name).sum()
    return float(wins / len(player_history))


def calculate_context_experience(
    player_name: str,
    history: pd.DataFrame,
    *,
    best_of: int | None = None,
    rounds: set[str] | None = None,
    surface: str | None = None,
    window: int = 30,
    scale: int = 8,
) -> float:
    player_history = _apply_context_filters(
        history,
        player_name,
        best_of=best_of,
        rounds=rounds,
        surface=surface,
        opponent_hand=None,
    )
    matches_in_context = len(player_history.tail(window))
    return float(min(matches_in_context / scale, 1.0))


def calculate_context_balance(
    player_name: str,
    history: pd.DataFrame,
    *,
    opponent_hand: str,
    window: int = 20,
    min_matches: int = 3,
    fallback: float = 0.0,
) -> float:
    player_history = _apply_context_filters(
        history,
        player_name,
        best_of=None,
        rounds=None,
        surface=None,
        opponent_hand=opponent_hand,
    ).tail(window)

    if len(player_history) < min_matches:
        return fallback

    wins = (player_history["winner_name"] == player_name).sum()
    losses = len(player_history) - wins
    return float((wins - losses) / len(player_history))


def add_targeted_slice_features(
    df_subset: pd.DataFrame,
    historical_data: pd.DataFrame,
    base_cols: list[str],
) -> pd.DataFrame:
    """Add raw winner/loser context features before symmetrization."""
    df_subset = df_subset.copy()

    full_sequence = pd.concat(
        [historical_data[base_cols], df_subset[base_cols]],
        ignore_index=True,
    )
    start_idx = len(historical_data)
    player_index = build_player_index(full_sequence)

    w_best_of5_form_list = []
    l_best_of5_form_list = []
    w_late_round_form_list = []
    l_late_round_form_list = []
    w_best_of5_experience_list = []
    l_best_of5_experience_list = []
    w_late_round_experience_list = []
    l_late_round_experience_list = []
    w_vs_opp_hand_form_list = []
    l_vs_opp_hand_form_list = []
    w_qf_form_list = []
    l_qf_form_list = []
    w_qf_experience_list = []
    l_qf_experience_list = []
    w_qf_surface_form_list = []
    l_qf_surface_form_list = []
    w_vs_opp_hand_surface_form_list = []
    l_vs_opp_hand_surface_form_list = []
    w_vs_opp_hand_balance_list = []
    l_vs_opp_hand_balance_list = []

    for i in range(len(df_subset)):
        row = df_subset.iloc[i]
        cutoff = start_idx + i

        winner_name = row["winner_name"]
        loser_name = row["loser_name"]
        surface = row["surface"]
        winner_form = float(row["w_form"])
        loser_form = float(row["l_form"])
        winner_surface_form = float(row["w_surface_form"])
        loser_surface_form = float(row["l_surface_form"])

        winner_history = get_player_history_via_index(
            winner_name, full_sequence, player_index, cutoff
        )
        loser_history = get_player_history_via_index(
            loser_name, full_sequence, player_index, cutoff
        )

        w_best_of5_form_list.append(
            calculate_context_form(
                winner_name,
                winner_history,
                best_of=5,
                window=8,
                min_matches=2,
                fallback=winner_form,
            )
        )
        l_best_of5_form_list.append(
            calculate_context_form(
                loser_name,
                loser_history,
                best_of=5,
                window=8,
                min_matches=2,
                fallback=loser_form,
            )
        )

        w_late_round_form_list.append(
            calculate_context_form(
                winner_name,
                winner_history,
                rounds=LATE_ROUNDS,
                window=8,
                min_matches=2,
                fallback=winner_form,
            )
        )
        l_late_round_form_list.append(
            calculate_context_form(
                loser_name,
                loser_history,
                rounds=LATE_ROUNDS,
                window=8,
                min_matches=2,
                fallback=loser_form,
            )
        )

        w_best_of5_experience_list.append(
            calculate_context_experience(
                winner_name,
                winner_history,
                best_of=5,
                window=20,
                scale=6,
            )
        )
        l_best_of5_experience_list.append(
            calculate_context_experience(
                loser_name,
                loser_history,
                best_of=5,
                window=20,
                scale=6,
            )
        )

        w_late_round_experience_list.append(
            calculate_context_experience(
                winner_name,
                winner_history,
                rounds=LATE_ROUNDS,
                window=20,
                scale=6,
            )
        )
        l_late_round_experience_list.append(
            calculate_context_experience(
                loser_name,
                loser_history,
                rounds=LATE_ROUNDS,
                window=20,
                scale=6,
            )
        )

        w_vs_opp_hand_form_list.append(
            calculate_context_form(
                winner_name,
                winner_history,
                opponent_hand=row["loser_hand"],
                window=12,
                min_matches=3,
                fallback=winner_form,
            )
        )
        l_vs_opp_hand_form_list.append(
            calculate_context_form(
                loser_name,
                loser_history,
                opponent_hand=row["winner_hand"],
                window=12,
                min_matches=3,
                fallback=loser_form,
            )
        )

        w_qf_form_list.append(
            calculate_context_form(
                winner_name,
                winner_history,
                rounds={"QF"},
                window=6,
                min_matches=1,
                fallback=winner_form,
            )
        )
        l_qf_form_list.append(
            calculate_context_form(
                loser_name,
                loser_history,
                rounds={"QF"},
                window=6,
                min_matches=1,
                fallback=loser_form,
            )
        )

        w_qf_experience_list.append(
            calculate_context_experience(
                winner_name,
                winner_history,
                rounds={"QF"},
                window=16,
                scale=4,
            )
        )
        l_qf_experience_list.append(
            calculate_context_experience(
                loser_name,
                loser_history,
                rounds={"QF"},
                window=16,
                scale=4,
            )
        )

        w_qf_surface_form_list.append(
            calculate_context_form(
                winner_name,
                winner_history,
                rounds={"QF"},
                surface=surface,
                window=4,
                min_matches=1,
                fallback=winner_surface_form,
            )
        )
        l_qf_surface_form_list.append(
            calculate_context_form(
                loser_name,
                loser_history,
                rounds={"QF"},
                surface=surface,
                window=4,
                min_matches=1,
                fallback=loser_surface_form,
            )
        )

        w_vs_opp_hand_surface_form_list.append(
            calculate_context_form(
                winner_name,
                winner_history,
                surface=surface,
                opponent_hand=row["loser_hand"],
                window=8,
                min_matches=2,
                fallback=winner_surface_form,
            )
        )
        l_vs_opp_hand_surface_form_list.append(
            calculate_context_form(
                loser_name,
                loser_history,
                surface=surface,
                opponent_hand=row["winner_hand"],
                window=8,
                min_matches=2,
                fallback=loser_surface_form,
            )
        )

        w_vs_opp_hand_balance_list.append(
            calculate_context_balance(
                winner_name,
                winner_history,
                opponent_hand=row["loser_hand"],
                window=20,
                min_matches=3,
                fallback=0.0,
            )
        )
        l_vs_opp_hand_balance_list.append(
            calculate_context_balance(
                loser_name,
                loser_history,
                opponent_hand=row["winner_hand"],
                window=20,
                min_matches=3,
                fallback=0.0,
            )
        )

    df_subset["w_best_of5_form"] = w_best_of5_form_list
    df_subset["l_best_of5_form"] = l_best_of5_form_list
    df_subset["w_late_round_form"] = w_late_round_form_list
    df_subset["l_late_round_form"] = l_late_round_form_list
    df_subset["w_best_of5_experience"] = w_best_of5_experience_list
    df_subset["l_best_of5_experience"] = l_best_of5_experience_list
    df_subset["w_late_round_experience"] = w_late_round_experience_list
    df_subset["l_late_round_experience"] = l_late_round_experience_list
    df_subset["w_vs_opp_hand_form"] = w_vs_opp_hand_form_list
    df_subset["l_vs_opp_hand_form"] = l_vs_opp_hand_form_list
    df_subset["w_qf_form"] = w_qf_form_list
    df_subset["l_qf_form"] = l_qf_form_list
    df_subset["w_qf_experience"] = w_qf_experience_list
    df_subset["l_qf_experience"] = l_qf_experience_list
    df_subset["w_qf_surface_form"] = w_qf_surface_form_list
    df_subset["l_qf_surface_form"] = l_qf_surface_form_list
    df_subset["w_vs_opp_hand_surface_form"] = w_vs_opp_hand_surface_form_list
    df_subset["l_vs_opp_hand_surface_form"] = l_vs_opp_hand_surface_form_list
    df_subset["w_vs_opp_hand_balance"] = w_vs_opp_hand_balance_list
    df_subset["l_vs_opp_hand_balance"] = l_vs_opp_hand_balance_list

    return df_subset


def attach_targeted_features(
    symmetrized_data: pd.DataFrame,
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    helper_columns = [
        "match_id",
        "round",
        "winner_hand",
        "loser_hand",
        "w_best_of5_form",
        "l_best_of5_form",
        "w_late_round_form",
        "l_late_round_form",
        "w_best_of5_experience",
        "l_best_of5_experience",
        "w_late_round_experience",
        "l_late_round_experience",
        "w_vs_opp_hand_form",
        "l_vs_opp_hand_form",
        "w_qf_form",
        "l_qf_form",
        "w_qf_experience",
        "l_qf_experience",
        "w_qf_surface_form",
        "l_qf_surface_form",
        "w_vs_opp_hand_surface_form",
        "l_vs_opp_hand_surface_form",
        "w_vs_opp_hand_balance",
        "l_vs_opp_hand_balance",
    ]
    enriched = symmetrized_data.merge(
        raw_data[helper_columns],
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    winner_perspective_mask = enriched["y"] == 1

    enriched["is_best_of5"] = (enriched["best_of"] == 5).astype(int)
    enriched["is_qf"] = (enriched["round"] == "QF").astype(int)
    enriched["is_lefty_matchup"] = (enriched["winner_hand"] != enriched["loser_hand"]).astype(int)

    enriched["p1_best_of5_form"] = np.where(
        winner_perspective_mask,
        enriched["w_best_of5_form"],
        enriched["l_best_of5_form"],
    )
    enriched["p2_best_of5_form"] = np.where(
        winner_perspective_mask,
        enriched["l_best_of5_form"],
        enriched["w_best_of5_form"],
    )
    enriched["best_of5_form_diff"] = (
        enriched["p1_best_of5_form"] - enriched["p2_best_of5_form"]
    )

    enriched["p1_best_of5_experience"] = np.where(
        winner_perspective_mask,
        enriched["w_best_of5_experience"],
        enriched["l_best_of5_experience"],
    )
    enriched["p2_best_of5_experience"] = np.where(
        winner_perspective_mask,
        enriched["l_best_of5_experience"],
        enriched["w_best_of5_experience"],
    )
    enriched["best_of5_experience_diff"] = (
        enriched["p1_best_of5_experience"] - enriched["p2_best_of5_experience"]
    )

    enriched["p1_late_round_form"] = np.where(
        winner_perspective_mask,
        enriched["w_late_round_form"],
        enriched["l_late_round_form"],
    )
    enriched["p2_late_round_form"] = np.where(
        winner_perspective_mask,
        enriched["l_late_round_form"],
        enriched["w_late_round_form"],
    )
    enriched["late_round_form_diff"] = (
        enriched["p1_late_round_form"] - enriched["p2_late_round_form"]
    )

    enriched["p1_late_round_experience"] = np.where(
        winner_perspective_mask,
        enriched["w_late_round_experience"],
        enriched["l_late_round_experience"],
    )
    enriched["p2_late_round_experience"] = np.where(
        winner_perspective_mask,
        enriched["l_late_round_experience"],
        enriched["w_late_round_experience"],
    )
    enriched["late_round_experience_diff"] = (
        enriched["p1_late_round_experience"] - enriched["p2_late_round_experience"]
    )

    enriched["p1_vs_opp_hand_form"] = np.where(
        winner_perspective_mask,
        enriched["w_vs_opp_hand_form"],
        enriched["l_vs_opp_hand_form"],
    )
    enriched["p2_vs_opp_hand_form"] = np.where(
        winner_perspective_mask,
        enriched["l_vs_opp_hand_form"],
        enriched["w_vs_opp_hand_form"],
    )
    enriched["opp_hand_form_diff"] = (
        enriched["p1_vs_opp_hand_form"] - enriched["p2_vs_opp_hand_form"]
    )

    enriched["p1_qf_form"] = np.where(
        winner_perspective_mask,
        enriched["w_qf_form"],
        enriched["l_qf_form"],
    )
    enriched["p2_qf_form"] = np.where(
        winner_perspective_mask,
        enriched["l_qf_form"],
        enriched["w_qf_form"],
    )
    enriched["qf_form_diff"] = enriched["p1_qf_form"] - enriched["p2_qf_form"]

    enriched["p1_qf_experience"] = np.where(
        winner_perspective_mask,
        enriched["w_qf_experience"],
        enriched["l_qf_experience"],
    )
    enriched["p2_qf_experience"] = np.where(
        winner_perspective_mask,
        enriched["l_qf_experience"],
        enriched["w_qf_experience"],
    )
    enriched["qf_experience_diff"] = (
        enriched["p1_qf_experience"] - enriched["p2_qf_experience"]
    )

    enriched["p1_qf_surface_form"] = np.where(
        winner_perspective_mask,
        enriched["w_qf_surface_form"],
        enriched["l_qf_surface_form"],
    )
    enriched["p2_qf_surface_form"] = np.where(
        winner_perspective_mask,
        enriched["l_qf_surface_form"],
        enriched["w_qf_surface_form"],
    )
    enriched["qf_surface_form_diff"] = (
        enriched["p1_qf_surface_form"] - enriched["p2_qf_surface_form"]
    )

    enriched["p1_vs_opp_hand_surface_form"] = np.where(
        winner_perspective_mask,
        enriched["w_vs_opp_hand_surface_form"],
        enriched["l_vs_opp_hand_surface_form"],
    )
    enriched["p2_vs_opp_hand_surface_form"] = np.where(
        winner_perspective_mask,
        enriched["l_vs_opp_hand_surface_form"],
        enriched["w_vs_opp_hand_surface_form"],
    )
    enriched["opp_hand_surface_form_diff"] = (
        enriched["p1_vs_opp_hand_surface_form"] - enriched["p2_vs_opp_hand_surface_form"]
    )

    enriched["p1_vs_opp_hand_balance"] = np.where(
        winner_perspective_mask,
        enriched["w_vs_opp_hand_balance"],
        enriched["l_vs_opp_hand_balance"],
    )
    enriched["p2_vs_opp_hand_balance"] = np.where(
        winner_perspective_mask,
        enriched["l_vs_opp_hand_balance"],
        enriched["w_vs_opp_hand_balance"],
    )
    enriched["opp_hand_balance_diff"] = (
        enriched["p1_vs_opp_hand_balance"] - enriched["p2_vs_opp_hand_balance"]
    )

    return enriched.drop(
        columns=[
            "round",
            "winner_hand",
            "loser_hand",
            "w_best_of5_form",
            "l_best_of5_form",
            "w_late_round_form",
            "l_late_round_form",
            "w_best_of5_experience",
            "l_best_of5_experience",
            "w_late_round_experience",
            "l_late_round_experience",
            "w_vs_opp_hand_form",
            "l_vs_opp_hand_form",
            "w_qf_form",
            "l_qf_form",
            "w_qf_experience",
            "l_qf_experience",
            "w_qf_surface_form",
            "l_qf_surface_form",
            "w_vs_opp_hand_surface_form",
            "l_vs_opp_hand_surface_form",
            "w_vs_opp_hand_balance",
            "l_vs_opp_hand_balance",
        ]
    )


def compute_symmetric_match_evaluation(test_data, threshold=0.5):
    """Match-level evaluation laczaca OBIE perspektywy symetryzowanego meczu.

    Kazdy mecz ma dwa wiersze o tym samym match_id: y==1 (p1=zwyciezca) i y==0
    (p1=przegrany). Prawdopodobienstwo wygranej RZECZYWISTEGO zwyciezcy usredniamy:
        z y==1: P_a = p1_win_probability       (p1 to zwyciezca)
        z y==0: P_b = 1 - p1_win_probability   (zwyciezca jest jako p2)
    winner_prob = (P_a + P_b) / 2; trafienie gdy winner_prob > threshold.
    Spojne z baseline -- metryka odporna na arbitralny labeling p1/p2.
    """
    winner_view = test_data[test_data["y"] == 1][
        ["match_id", "p1_name", "p2_name", "actual_winner", "p1_win_probability"]
    ].copy()
    loser_view = test_data[test_data["y"] == 0][["match_id", "p1_win_probability"]].rename(
        columns={"p1_win_probability": "loser_view_p1_prob"}
    )
    merged = winner_view.merge(loser_view, on="match_id", validate="one_to_one")
    merged["p1_win_probability"] = (
        merged["p1_win_probability"] + (1.0 - merged["loser_view_p1_prob"])
    ) / 2.0
    merged = merged.drop(columns=["loser_view_p1_prob"])
    merged["predicted_winner"] = np.where(
        merged["p1_win_probability"] > threshold, merged["p1_name"], merged["p2_name"]
    )
    merged["correct_prediction"] = merged["p1_win_probability"] > threshold
    accuracy = float(merged["correct_prediction"].mean())
    return merged, accuracy


def print_metric_delta(name: str, baseline_value: float, new_value: float) -> None:
    delta = new_value - baseline_value
    print(
        f"{name:<18} baseline={baseline_value:.4f} | slice-aware={new_value:.4f} "
        f"| delta={delta:+.4f}"
    )


def run_sliceaware_model() -> None:
    global search
    global features
    global best_rf
    global df_train_raw
    global df_val_raw
    global df_test_raw
    global val_data
    global test_data
    global winner_perspective
    global match_accuracy
    global val_acc
    global test_acc
    global feature_importance

    namespace = execute_base_pipeline_quietly()
    base_cols = list(namespace["cols_base"])
    symmetrize_data = namespace["symmetrize_data"]
    baseline_search = namespace["search"]

    baseline_val_acc = float(namespace["val_acc"])
    baseline_test_acc = float(namespace["test_acc"])
    baseline_match_accuracy = float(namespace["match_accuracy"])

    df_history_base = namespace["df_history_base"].copy()
    df_train_raw = add_targeted_slice_features(
        namespace["df_train_raw"].copy(),
        df_history_base,
        base_cols,
    )

    history_val = pd.concat(
        [df_history_base, df_train_raw[base_cols]],
        ignore_index=True,
    )
    df_val_raw = add_targeted_slice_features(
        namespace["df_val_raw"].copy(),
        history_val,
        base_cols,
    )

    history_test = pd.concat(
        [df_history_base, df_train_raw[base_cols], df_val_raw[base_cols]],
        ignore_index=True,
    )
    df_test_raw = add_targeted_slice_features(
        namespace["df_test_raw"].copy(),
        history_test,
        base_cols,
    )

    train_data_final = attach_targeted_features(symmetrize_data(df_train_raw, shuffle=True), df_train_raw)
    val_data = attach_targeted_features(symmetrize_data(df_val_raw, shuffle=True), df_val_raw)
    test_data = attach_targeted_features(symmetrize_data(df_test_raw, shuffle=True), df_test_raw)

    features = list(namespace["features"]) + TARGETED_FEATURES

    X_train_final = train_data_final[features]
    y_train_final = train_data_final["y"]
    X_val = val_data[features]
    y_val = val_data["y"]
    X_test = test_data[features]
    y_test = test_data["y"]

    best_rf = RandomForestClassifier(
        **baseline_search.best_params_,
        n_jobs=-1,
        random_state=namespace["RANDOM_STATE"],
    )

    print("=" * 70)
    print("SLICE-AWARE FEATURE EXTENSION")
    print("=" * 70)
    print("Nowe cechy v2: Best of 5, dokladny QF, QF na nawierzchni i bilans pod matchup reki")
    print("Model uzywa tych samych tuned hyperparameters co baseline.")
    print(f"Liczba cech: {len(features)} (baseline: {len(namespace['features'])})")
    print()

    best_rf.fit(X_train_final, y_train_final)

    val_pred = best_rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print("=== WALIDACJA ===")
    print(f"Accuracy: {val_acc:.4f}")
    print(classification_report(y_val, val_pred, target_names=["Gracz 2 wygrywa", "Gracz 1 wygrywa"]))
    print("Macierz pomylek:")
    print(confusion_matrix(y_val, val_pred))
    print()

    test_pred = best_rf.predict(X_test)
    test_pred_proba = best_rf.predict_proba(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print("=== TEST ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, test_pred, target_names=["Gracz 2 wygrywa", "Gracz 1 wygrywa"]))
    print("Macierz pomylek:")
    print(confusion_matrix(y_test, test_pred))
    print()

    test_data["p1_win_probability"] = test_pred_proba[:, 1]
    winner_perspective, match_accuracy = compute_symmetric_match_evaluation(test_data)

    print("=== MATCH-LEVEL ===")
    print(f"Accuracy przewidywania zwyciezcow: {match_accuracy:.4f} ({match_accuracy * 100:.2f}%)")
    print(
        f"Poprawnie przewidziane: {int(winner_perspective['correct_prediction'].sum())} / "
        f"{len(winner_perspective)} meczow"
    )
    print()

    print("=== POROWNANIE Z BASELINE ===")
    print_metric_delta("Validation", baseline_val_acc, val_acc)
    print_metric_delta("Test", baseline_test_acc, test_acc)
    print_metric_delta("Match-level", baseline_match_accuracy, match_accuracy)
    print()

    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": best_rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("=== NOWE CECHY: WAZNOSC ===")
    print(
        feature_importance[
            feature_importance["feature"].isin(TARGETED_FEATURES)
        ].to_string(index=False)
    )

    search = baseline_search


run_sliceaware_model()