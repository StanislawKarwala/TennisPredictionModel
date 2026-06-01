"""
Slice-aware extension of main_48_cech.py with QF v3 and serve v2 features.

QF v3 focuses on:
  - tournament level pressure,
  - seed context / seed proxy,
  - strength of opponents already faced in the same tournament.

Serve v2 focuses on contextual serve and return quality:
  - on the current surface,
  - in pressure contexts (Best of 5 / late rounds),
  - against strong opponents,
  - against the current opponent handedness.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --- Player history index (inlined) -----------------------------------------
# qfserve_v3 ma ~25 wywolan filter_player_history na mecz. Bez indexu kazde
# wymagaloby skanu pelnej historii ~18-20k wierszy. Index buduje raz strukture
# `player_name -> posortowane indeksy wierszy` i potem lookup w O(log K).

class PlayerHistoryIndex:
    """Mapa `player_name -> posortowane indeksy wierszy w full_sequence`."""

    __slots__ = ("_full_sequence", "_player_to_indices")

    def __init__(self, full_sequence: pd.DataFrame) -> None:
        self._full_sequence = full_sequence
        if not full_sequence.index.equals(pd.RangeIndex(len(full_sequence))):
            raise ValueError(
                "PlayerHistoryIndex wymaga full_sequence z range index 0..N-1."
            )
        winner_names = full_sequence["winner_name"].to_numpy()
        loser_names = full_sequence["loser_name"].to_numpy()
        row_indices = np.arange(len(full_sequence))
        combined = pd.concat([
            pd.Series(row_indices, index=winner_names),
            pd.Series(row_indices, index=loser_names),
        ])
        self._player_to_indices: dict[str, np.ndarray] = {
            player: np.sort(group.to_numpy())
            for player, group in combined.groupby(level=0)
        }

    def past_for(self, player: str, exclusive_end: int) -> pd.DataFrame:
        """Mecze gracza rozegrane scisle wczesniej niz exclusive_end (chronologicznie)."""
        indices = self._player_to_indices.get(player)
        if indices is None:
            return self._full_sequence.iloc[0:0]
        cutoff = np.searchsorted(indices, exclusive_end, side="left")
        if cutoff == 0:
            return self._full_sequence.iloc[0:0]
        return self._full_sequence.iloc[indices[:cutoff]]


BASE_SCRIPT = Path(__file__).with_name("main_48_cech.py")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample_data"
TOUR = os.environ.get("TENNIS_TOUR", "atp")
TARGET_YEAR = int(os.environ.get("TENNIS_TARGET_YEAR", "2025"))
HISTORY_START_YEAR = int(os.environ.get("TENNIS_HISTORY_START", "2001"))


def data_file(year: int) -> Path:
    return DATA_DIR / f"{TOUR}_matches_{year}.csv"


TOURNEY_ID_PATTERN = re.compile(r"^\d{4}-")
HISTORY_FILES = [data_file(y) for y in range(HISTORY_START_YEAR, TARGET_YEAR)]
PRESSURE_ROUNDS = {"QF", "SF", "BR", "F"}
LATE_ROUNDS = PRESSURE_ROUNDS
EXTRA_CONTEXT_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "draw_size",
    "winner_seed",
    "loser_seed",
]
TOURNEY_LEVEL_STRENGTH = {
    "G": 1.00,
    "M": 0.92,
    "F": 0.88,
    "A": 0.84,
    "500": 0.78,
    "250": 0.68,
    "D": 0.58,
    "O": 0.50,
}

TARGETED_FEATURES = [
    "is_best_of5",
    "is_qf",
    "is_lefty_matchup",
    "tourney_level_strength",
    "qf_level_pressure",
    "best_of5_level_pressure",
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
    "p1_seed_context_score",
    "p2_seed_context_score",
    "seed_context_diff",
    "p1_tourney_path_opp_strength",
    "p2_tourney_path_opp_strength",
    "tourney_path_opp_strength_diff",
    "p1_tourney_path_match_count",
    "p2_tourney_path_match_count",
    "tourney_path_match_count_diff",
    "p1_surface_serve_score",
    "p2_surface_serve_score",
    "surface_serve_score_diff",
    "p1_top_opp_serve_score",
    "p2_top_opp_serve_score",
    "top_opp_serve_score_diff",
    "p1_vs_opp_hand_return_score",
    "p2_vs_opp_hand_return_score",
    "vs_opp_hand_return_score_diff",
    "p1_surface_serve_stability",
    "p2_surface_serve_stability",
    "surface_serve_stability_diff",
    "p1_pressure_serve_score",
    "p2_pressure_serve_score",
    "pressure_serve_score_diff",
]

SYMMETRIC_FEATURE_SPECS = [
    ("best_of5_form", "best_of5_form_diff"),
    ("best_of5_experience", "best_of5_experience_diff"),
    ("late_round_form", "late_round_form_diff"),
    ("late_round_experience", "late_round_experience_diff"),
    ("vs_opp_hand_form", "opp_hand_form_diff"),
    ("qf_form", "qf_form_diff"),
    ("qf_experience", "qf_experience_diff"),
    ("qf_surface_form", "qf_surface_form_diff"),
    ("vs_opp_hand_surface_form", "opp_hand_surface_form_diff"),
    ("vs_opp_hand_balance", "opp_hand_balance_diff"),
    ("seed_context_score", "seed_context_diff"),
    ("tourney_path_opp_strength", "tourney_path_opp_strength_diff"),
    ("tourney_path_match_count", "tourney_path_match_count_diff"),
    ("surface_serve_score", "surface_serve_score_diff"),
    ("top_opp_serve_score", "top_opp_serve_score_diff"),
    ("vs_opp_hand_return_score", "vs_opp_hand_return_score_diff"),
    ("surface_serve_stability", "surface_serve_stability_diff"),
    ("pressure_serve_score", "pressure_serve_score_diff"),
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


def unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def load_context_frame(csv_path, base_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    keep_columns = unique_columns(base_cols + EXTRA_CONTEXT_COLUMNS)
    frame = df[keep_columns].dropna(subset=base_cols).reset_index(drop=True)
    validate_tourney_id_format(frame, source=csv_path)
    return frame


def validate_tourney_id_format(frame: pd.DataFrame, *, source: str) -> None:
    if "tourney_id" not in frame.columns:
        return
    tourney_ids = frame["tourney_id"].dropna().astype(str)
    invalid = [tid for tid in tourney_ids.unique() if not TOURNEY_ID_PATTERN.match(tid)]
    if invalid:
        raise ValueError(
            f"tourney_id w {source} nie ma formatu YYYY-XXX, "
            f"co grozi leakage w calculate_tournament_path_stats. "
            f"Przykladowe zle ID: {invalid[:5]}"
        )


def load_context_data(
    base_cols: list[str],
    train_len: int,
    val_len: int,
    test_len: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_2024 = load_context_frame(data_file(TARGET_YEAR), base_cols)
    expected_len = train_len + val_len + test_len
    if len(df_2024) != expected_len:
        raise ValueError(
            "Niespojnosc przy ladowaniu danych kontekstowych 2024: "
            f"{len(df_2024)} vs expected {expected_len}."
        )

    history_parts = [load_context_frame(path, base_cols) for path in HISTORY_FILES]
    history_context = pd.concat(history_parts, ignore_index=True)

    train_context = df_2024.iloc[:train_len].reset_index(drop=True)
    val_context = df_2024.iloc[train_len:train_len + val_len].reset_index(drop=True)
    test_context = df_2024.iloc[train_len + val_len:].reset_index(drop=True)

    for frame in (train_context, val_context, test_context):
        frame["match_id"] = range(len(frame))

    return history_context, train_context, val_context, test_context


def attach_context_columns(raw_split: pd.DataFrame, context_split: pd.DataFrame) -> pd.DataFrame:
    return raw_split.merge(
        context_split[["match_id"] + EXTRA_CONTEXT_COLUMNS],
        on="match_id",
        how="left",
        validate="one_to_one",
    )


# Module-level kontekst dla per-row indeksu: set_history_context() jest
# wywolywany RAZ na iteracje w add_targeted_slice_features, dzieki czemu
# get_player_history nie musi skanowac calego pandasa (filter_player_history
# robil to do 25 razy na mecz w v3).
_HISTORY_INDEX: "PlayerHistoryIndex | None" = None
_HISTORY_CUTOFF: int | None = None


def set_history_context(index: "PlayerHistoryIndex | None", cutoff: int | None) -> None:
    global _HISTORY_INDEX, _HISTORY_CUTOFF
    _HISTORY_INDEX = index
    _HISTORY_CUTOFF = cutoff


def get_player_history(player_name: str, history: pd.DataFrame) -> pd.DataFrame:
    if _HISTORY_INDEX is not None and _HISTORY_CUTOFF is not None:
        return _HISTORY_INDEX.past_for(player_name, _HISTORY_CUTOFF)
    return history[
        (history["winner_name"] == player_name) |
        (history["loser_name"] == player_name)
    ]


def filter_player_history(
    player_name: str,
    history: pd.DataFrame,
    *,
    best_of: int | None = None,
    rounds: set[str] | None = None,
    surface: str | None = None,
    opponent_hand: str | None = None,
    opponent_rank_max: int | None = None,
) -> pd.DataFrame:
    player_history = get_player_history(player_name, history)

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

    if opponent_rank_max is not None:
        top_opp_mask = (
            ((player_history["winner_name"] == player_name) & (player_history["loser_rank"] <= opponent_rank_max)) |
            ((player_history["loser_name"] == player_name) & (player_history["winner_rank"] <= opponent_rank_max))
        )
        player_history = player_history[top_opp_mask]

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
    player_history = filter_player_history(
        player_name,
        history,
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
    player_history = filter_player_history(
        player_name,
        history,
        best_of=best_of,
        rounds=rounds,
        surface=surface,
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
    player_history = filter_player_history(
        player_name,
        history,
        opponent_hand=opponent_hand,
    ).tail(window)

    if len(player_history) < min_matches:
        return fallback

    wins = (player_history["winner_name"] == player_name).sum()
    losses = len(player_history) - wins
    return float((wins - losses) / len(player_history))


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0 else default


def extract_player_match_serve_metrics(match: pd.Series, player_name: str) -> dict[str, float]:
    is_winner = match["winner_name"] == player_name

    if is_winner:
        svpt = float(match["w_svpt"])
        ace = float(match["w_ace"])
        df = float(match["w_df"])
        first_in = float(match["w_1stIn"])
        first_won = float(match["w_1stWon"])
        second_won = float(match["w_2ndWon"])
        sv_gms = float(match["w_SvGms"])
        bp_saved = float(match["w_bpSaved"])
        bp_faced = float(match["w_bpFaced"])
        opp_svpt = float(match["l_svpt"])
        opp_first_won = float(match["l_1stWon"])
        opp_second_won = float(match["l_2ndWon"])
    else:
        svpt = float(match["l_svpt"])
        ace = float(match["l_ace"])
        df = float(match["l_df"])
        first_in = float(match["l_1stIn"])
        first_won = float(match["l_1stWon"])
        second_won = float(match["l_2ndWon"])
        sv_gms = float(match["l_SvGms"])
        bp_saved = float(match["l_bpSaved"])
        bp_faced = float(match["l_bpFaced"])
        opp_svpt = float(match["w_svpt"])
        opp_first_won = float(match["w_1stWon"])
        opp_second_won = float(match["w_2ndWon"])

    second_serve_points = max(svpt - first_in, 0.0)
    return {
        "ace_rate": safe_ratio(ace, svpt, 0.08),
        "df_rate": safe_ratio(df, svpt, 0.03),
        "first_in_pct": safe_ratio(first_in, svpt, 0.60),
        "first_won_pct": safe_ratio(first_won, first_in, 0.70),
        "second_won_pct": safe_ratio(second_won, second_serve_points, 0.50),
        "bp_save_pct": safe_ratio(bp_saved, bp_faced, 0.60),
        "bp_faced_per_game": safe_ratio(bp_faced, sv_gms, 0.40),
        "return_pts_won": safe_ratio(
            opp_svpt - opp_first_won - opp_second_won,
            opp_svpt,
            0.35,
        ),
    }


def compose_serve_score(stats: dict[str, float]) -> float:
    ace_component = min(stats["ace_rate"] / 0.15, 1.5)
    df_component = min(stats["df_rate"] / 0.08, 1.5)
    bp_faced_component = min(stats["bp_faced_per_game"] / 0.80, 1.5)

    return float(
        0.10 * ace_component
        + 0.18 * stats["first_in_pct"]
        + 0.24 * stats["first_won_pct"]
        + 0.22 * stats["second_won_pct"]
        + 0.14 * stats["bp_save_pct"]
        + 0.12 * stats["return_pts_won"]
        - 0.08 * df_component
        - 0.08 * bp_faced_component
    )


def build_fallback_serve_profile(row: pd.Series, prefix: str) -> dict[str, float]:
    stats = {
        "ace_rate": float(row[f"{prefix}_ace_rate"]),
        "df_rate": float(row[f"{prefix}_df_rate"]),
        "first_in_pct": float(row[f"{prefix}_first_in_pct"]),
        "first_won_pct": float(row[f"{prefix}_first_won_pct"]),
        "second_won_pct": float(row[f"{prefix}_second_won_pct"]),
        "bp_save_pct": float(row[f"{prefix}_bp_save_pct"]),
        "bp_faced_per_game": float(row[f"{prefix}_bp_faced_per_game"]),
        "return_pts_won": float(row[f"{prefix}_return_pts_won"]),
    }
    return {
        "serve_score": compose_serve_score(stats),
        "return_score": stats["return_pts_won"],
        "stability": 0.50,
    }


def calculate_context_serve_profile(
    player_name: str,
    history: pd.DataFrame,
    *,
    best_of: int | None = None,
    rounds: set[str] | None = None,
    surface: str | None = None,
    opponent_hand: str | None = None,
    opponent_rank_max: int | None = None,
    window: int = 10,
    min_matches: int = 2,
    fallback: dict[str, float] | None = None,
) -> dict[str, float]:
    if fallback is None:
        fallback = {"serve_score": 0.55, "return_score": 0.35, "stability": 0.50}

    player_history = filter_player_history(
        player_name,
        history,
        best_of=best_of,
        rounds=rounds,
        surface=surface,
        opponent_hand=opponent_hand,
        opponent_rank_max=opponent_rank_max,
    ).tail(window)

    if len(player_history) < min_matches:
        return fallback

    match_metrics = [
        extract_player_match_serve_metrics(match, player_name)
        for _, match in player_history.iterrows()
    ]
    serve_scores = [compose_serve_score(metrics) for metrics in match_metrics]

    return {
        "serve_score": float(np.mean(serve_scores)),
        "return_score": float(np.mean([metrics["return_pts_won"] for metrics in match_metrics])),
        "stability": float(1.0 / (1.0 + np.std(serve_scores))),
    }


def estimate_seed_slots(draw_size: object) -> int:
    draw_size_value = pd.to_numeric(pd.Series([draw_size]), errors="coerce").iloc[0]
    if pd.isna(draw_size_value):
        return 8
    if draw_size_value >= 96:
        return 32
    if draw_size_value >= 56:
        return 16
    if draw_size_value >= 28:
        return 8
    return 4


def compute_seed_context_score(seed_value: object, rank: float, draw_size: object) -> float:
    seed_slots = estimate_seed_slots(draw_size)
    numeric_seed = pd.to_numeric(pd.Series([seed_value]), errors="coerce").iloc[0]

    if not pd.isna(numeric_seed):
        return float(max(0.0, 1.0 - (numeric_seed - 1.0) / seed_slots))

    soft_proxy = (seed_slots + 2.0 - float(rank)) / (seed_slots + 2.0)
    return float(np.clip(soft_proxy, 0.0, 1.0))


def strong_opponent_threshold(tourney_level: str) -> int:
    if tourney_level in {"G", "M"}:
        return 20
    if tourney_level in {"500", "A", "F"}:
        return 30
    return 40


def tournament_level_strength(tourney_level: str) -> float:
    return float(TOURNEY_LEVEL_STRENGTH.get(tourney_level, 0.60))


def opponent_rank_points(match: pd.Series, player_name: str) -> float:
    if match["winner_name"] == player_name:
        return float(match["loser_rank_points"])
    return float(match["winner_rank_points"])


def calculate_tournament_path_stats(
    player_name: str,
    current_row: pd.Series,
    past_matches: pd.DataFrame,
) -> dict[str, float]:
    # UWAGA: filtrujemy po nazwisku BEZPOSREDNIO na same_tournament, a NIE przez
    # get_player_history. get_player_history korzysta z globalnego _HISTORY_INDEX,
    # ktory ignoruje przekazana ramke i zwraca cala kariere gracza -- przez to
    # cechy tourney_path_* liczyly sie na calej historii zamiast na biezacym
    # turnieju (cecha nie niosla sygnalu "trudnosci drogi"). past_matches jest
    # juz scisle chronologicznie wczesniejsze (full_sequence.iloc[:cutoff]),
    # a sortowanie tourney_date+match_num gwarantuje, ze widac tylko wczesniejsze
    # rundy tego samego turnieju.
    same_tournament = past_matches[past_matches["tourney_id"] == current_row["tourney_id"]]
    player_path = same_tournament[
        (same_tournament["winner_name"] == player_name)
        | (same_tournament["loser_name"] == player_name)
    ]

    if len(player_path) == 0:
        return {"opp_strength": 0.0, "match_count": 0.0}

    opponent_strengths = [
        np.log1p(max(opponent_rank_points(match, player_name), 1.0))
        for _, match in player_path.iterrows()
    ]
    return {
        "opp_strength": float(np.mean(opponent_strengths)),
        "match_count": float(len(player_path)),
    }


def pressure_serve_profile(
    player_name: str,
    row: pd.Series,
    history: pd.DataFrame,
    fallback: dict[str, float],
) -> dict[str, float]:
    if int(row["best_of"]) == 5:
        return calculate_context_serve_profile(
            player_name,
            history,
            best_of=5,
            window=8,
            min_matches=2,
            fallback=fallback,
        )

    if row["round"] in PRESSURE_ROUNDS:
        return calculate_context_serve_profile(
            player_name,
            history,
            rounds=PRESSURE_ROUNDS,
            window=8,
            min_matches=2,
            fallback=fallback,
        )

    return fallback


def add_targeted_slice_features(
    df_subset: pd.DataFrame,
    historical_data: pd.DataFrame,
    context_base_cols: list[str],
) -> pd.DataFrame:
    df_subset = df_subset.copy()

    full_sequence = pd.concat(
        [historical_data[context_base_cols], df_subset[context_base_cols]],
        ignore_index=True,
    )
    start_idx = len(historical_data)

    # qfserve_v3 ma ~25 wywolan filter_player_history na mecz -- bez indeksu
    # kazde z nich skanowalo 18-20k wierszy pelnej historii.
    history_index = PlayerHistoryIndex(full_sequence)

    feature_rows: list[dict[str, float]] = []
    for i in range(len(df_subset)):
        row = df_subset.iloc[i]
        cutoff = start_idx + i
        set_history_context(history_index, cutoff)
        past_matches = full_sequence.iloc[:cutoff]

        winner_name = row["winner_name"]
        loser_name = row["loser_name"]
        surface = row["surface"]
        winner_form = float(row["w_form"])
        loser_form = float(row["l_form"])
        winner_surface_form = float(row["w_surface_form"])
        loser_surface_form = float(row["l_surface_form"])

        winner_serve_fallback = build_fallback_serve_profile(row, "w")
        loser_serve_fallback = build_fallback_serve_profile(row, "l")
        top_opp_threshold = strong_opponent_threshold(row["tourney_level"])

        winner_path_stats = calculate_tournament_path_stats(winner_name, row, past_matches)
        loser_path_stats = calculate_tournament_path_stats(loser_name, row, past_matches)

        winner_surface_serve = calculate_context_serve_profile(
            winner_name,
            past_matches,
            surface=surface,
            window=8,
            min_matches=2,
            fallback=winner_serve_fallback,
        )
        loser_surface_serve = calculate_context_serve_profile(
            loser_name,
            past_matches,
            surface=surface,
            window=8,
            min_matches=2,
            fallback=loser_serve_fallback,
        )

        winner_top_opp_serve = calculate_context_serve_profile(
            winner_name,
            past_matches,
            opponent_rank_max=top_opp_threshold,
            window=8,
            min_matches=2,
            fallback=winner_surface_serve,
        )
        loser_top_opp_serve = calculate_context_serve_profile(
            loser_name,
            past_matches,
            opponent_rank_max=top_opp_threshold,
            window=8,
            min_matches=2,
            fallback=loser_surface_serve,
        )

        winner_vs_opp_hand_serve = calculate_context_serve_profile(
            winner_name,
            past_matches,
            opponent_hand=row["loser_hand"],
            window=10,
            min_matches=2,
            fallback=winner_surface_serve,
        )
        loser_vs_opp_hand_serve = calculate_context_serve_profile(
            loser_name,
            past_matches,
            opponent_hand=row["winner_hand"],
            window=10,
            min_matches=2,
            fallback=loser_surface_serve,
        )

        winner_pressure_serve = pressure_serve_profile(
            winner_name,
            row,
            past_matches,
            winner_surface_serve,
        )
        loser_pressure_serve = pressure_serve_profile(
            loser_name,
            row,
            past_matches,
            loser_surface_serve,
        )

        feature_rows.append(
            {
                "w_best_of5_form": calculate_context_form(
                    winner_name,
                    past_matches,
                    best_of=5,
                    window=8,
                    min_matches=2,
                    fallback=winner_form,
                ),
                "l_best_of5_form": calculate_context_form(
                    loser_name,
                    past_matches,
                    best_of=5,
                    window=8,
                    min_matches=2,
                    fallback=loser_form,
                ),
                "w_best_of5_experience": calculate_context_experience(
                    winner_name,
                    past_matches,
                    best_of=5,
                    window=20,
                    scale=6,
                ),
                "l_best_of5_experience": calculate_context_experience(
                    loser_name,
                    past_matches,
                    best_of=5,
                    window=20,
                    scale=6,
                ),
                "w_late_round_form": calculate_context_form(
                    winner_name,
                    past_matches,
                    rounds=LATE_ROUNDS,
                    window=8,
                    min_matches=2,
                    fallback=winner_form,
                ),
                "l_late_round_form": calculate_context_form(
                    loser_name,
                    past_matches,
                    rounds=LATE_ROUNDS,
                    window=8,
                    min_matches=2,
                    fallback=loser_form,
                ),
                "w_late_round_experience": calculate_context_experience(
                    winner_name,
                    past_matches,
                    rounds=LATE_ROUNDS,
                    window=20,
                    scale=6,
                ),
                "l_late_round_experience": calculate_context_experience(
                    loser_name,
                    past_matches,
                    rounds=LATE_ROUNDS,
                    window=20,
                    scale=6,
                ),
                "w_vs_opp_hand_form": calculate_context_form(
                    winner_name,
                    past_matches,
                    opponent_hand=row["loser_hand"],
                    window=12,
                    min_matches=3,
                    fallback=winner_form,
                ),
                "l_vs_opp_hand_form": calculate_context_form(
                    loser_name,
                    past_matches,
                    opponent_hand=row["winner_hand"],
                    window=12,
                    min_matches=3,
                    fallback=loser_form,
                ),
                "w_qf_form": calculate_context_form(
                    winner_name,
                    past_matches,
                    rounds={"QF"},
                    window=6,
                    min_matches=1,
                    fallback=winner_form,
                ),
                "l_qf_form": calculate_context_form(
                    loser_name,
                    past_matches,
                    rounds={"QF"},
                    window=6,
                    min_matches=1,
                    fallback=loser_form,
                ),
                "w_qf_experience": calculate_context_experience(
                    winner_name,
                    past_matches,
                    rounds={"QF"},
                    window=16,
                    scale=4,
                ),
                "l_qf_experience": calculate_context_experience(
                    loser_name,
                    past_matches,
                    rounds={"QF"},
                    window=16,
                    scale=4,
                ),
                "w_qf_surface_form": calculate_context_form(
                    winner_name,
                    past_matches,
                    rounds={"QF"},
                    surface=surface,
                    window=4,
                    min_matches=1,
                    fallback=winner_surface_form,
                ),
                "l_qf_surface_form": calculate_context_form(
                    loser_name,
                    past_matches,
                    rounds={"QF"},
                    surface=surface,
                    window=4,
                    min_matches=1,
                    fallback=loser_surface_form,
                ),
                "w_vs_opp_hand_surface_form": calculate_context_form(
                    winner_name,
                    past_matches,
                    surface=surface,
                    opponent_hand=row["loser_hand"],
                    window=8,
                    min_matches=2,
                    fallback=winner_surface_form,
                ),
                "l_vs_opp_hand_surface_form": calculate_context_form(
                    loser_name,
                    past_matches,
                    surface=surface,
                    opponent_hand=row["winner_hand"],
                    window=8,
                    min_matches=2,
                    fallback=loser_surface_form,
                ),
                "w_vs_opp_hand_balance": calculate_context_balance(
                    winner_name,
                    past_matches,
                    opponent_hand=row["loser_hand"],
                    window=20,
                    min_matches=3,
                    fallback=0.0,
                ),
                "l_vs_opp_hand_balance": calculate_context_balance(
                    loser_name,
                    past_matches,
                    opponent_hand=row["winner_hand"],
                    window=20,
                    min_matches=3,
                    fallback=0.0,
                ),
                "w_seed_context_score": compute_seed_context_score(
                    row["winner_seed"],
                    float(row["winner_rank"]),
                    row["draw_size"],
                ),
                "l_seed_context_score": compute_seed_context_score(
                    row["loser_seed"],
                    float(row["loser_rank"]),
                    row["draw_size"],
                ),
                "w_tourney_path_opp_strength": winner_path_stats["opp_strength"],
                "l_tourney_path_opp_strength": loser_path_stats["opp_strength"],
                "w_tourney_path_match_count": winner_path_stats["match_count"],
                "l_tourney_path_match_count": loser_path_stats["match_count"],
                "w_surface_serve_score": winner_surface_serve["serve_score"],
                "l_surface_serve_score": loser_surface_serve["serve_score"],
                "w_top_opp_serve_score": winner_top_opp_serve["serve_score"],
                "l_top_opp_serve_score": loser_top_opp_serve["serve_score"],
                "w_vs_opp_hand_return_score": winner_vs_opp_hand_serve["return_score"],
                "l_vs_opp_hand_return_score": loser_vs_opp_hand_serve["return_score"],
                "w_surface_serve_stability": winner_surface_serve["stability"],
                "l_surface_serve_stability": loser_surface_serve["stability"],
                "w_pressure_serve_score": winner_pressure_serve["serve_score"],
                "l_pressure_serve_score": loser_pressure_serve["serve_score"],
                "tourney_level_raw": row["tourney_level"],
            }
        )

    # Czyscimy module-level state, zeby nie wyciekal na kolejne wywolania
    # (np. add_targeted_slice_features dla trening -> walidacja -> test).
    set_history_context(None, None)

    feature_frame = pd.DataFrame(feature_rows)
    return pd.concat([df_subset.reset_index(drop=True), feature_frame], axis=1)


def attach_targeted_features(
    symmetrized_data: pd.DataFrame,
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    helper_columns = [
        "match_id",
        "round",
        "winner_hand",
        "loser_hand",
        "tourney_level_raw",
    ]
    for feature_name, _ in SYMMETRIC_FEATURE_SPECS:
        helper_columns.extend([f"w_{feature_name}", f"l_{feature_name}"])

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
    enriched["tourney_level_strength"] = (
        enriched["tourney_level_raw"].map(TOURNEY_LEVEL_STRENGTH).fillna(0.60).astype(float)
    )
    enriched["qf_level_pressure"] = enriched["is_qf"] * enriched["tourney_level_strength"]
    enriched["best_of5_level_pressure"] = enriched["is_best_of5"] * enriched["tourney_level_strength"]

    for feature_name, diff_name in SYMMETRIC_FEATURE_SPECS:
        winner_column = f"w_{feature_name}"
        loser_column = f"l_{feature_name}"
        p1_column = f"p1_{feature_name}"
        p2_column = f"p2_{feature_name}"

        enriched[p1_column] = np.where(
            winner_perspective_mask,
            enriched[winner_column],
            enriched[loser_column],
        )
        enriched[p2_column] = np.where(
            winner_perspective_mask,
            enriched[loser_column],
            enriched[winner_column],
        )
        enriched[diff_name] = enriched[p1_column] - enriched[p2_column]

    drop_columns = [
        "round",
        "winner_hand",
        "loser_hand",
        "tourney_level_raw",
    ]
    for feature_name, _ in SYMMETRIC_FEATURE_SPECS:
        drop_columns.extend([f"w_{feature_name}", f"l_{feature_name}"])

    return enriched.drop(columns=drop_columns)


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
        f"{name:<18} baseline={baseline_value:.4f} | qfserve-v3={new_value:.4f} "
        f"| delta={delta:+.4f}"
    )


def run_sliceaware_qfserve_v3() -> None:
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
    context_base_cols = unique_columns(base_cols + EXTRA_CONTEXT_COLUMNS)
    symmetrize_data = namespace["symmetrize_data"]
    baseline_search = namespace["search"]

    baseline_val_acc = float(namespace["val_acc"])
    baseline_test_acc = float(namespace["test_acc"])
    baseline_match_accuracy = float(namespace["match_accuracy"])

    history_context, train_context, val_context, test_context = load_context_data(
        base_cols,
        train_len=len(namespace["df_train_raw"]),
        val_len=len(namespace["df_val_raw"]),
        test_len=len(namespace["df_test_raw"]),
    )

    df_train_raw = attach_context_columns(namespace["df_train_raw"].copy(), train_context)
    df_val_raw = attach_context_columns(namespace["df_val_raw"].copy(), val_context)
    df_test_raw = attach_context_columns(namespace["df_test_raw"].copy(), test_context)

    df_train_raw = add_targeted_slice_features(df_train_raw, history_context, context_base_cols)

    history_val = pd.concat(
        [history_context, df_train_raw[context_base_cols]],
        ignore_index=True,
    )
    df_val_raw = add_targeted_slice_features(df_val_raw, history_val, context_base_cols)

    history_test = pd.concat(
        [history_context, df_train_raw[context_base_cols], df_val_raw[context_base_cols]],
        ignore_index=True,
    )
    df_test_raw = add_targeted_slice_features(df_test_raw, history_test, context_base_cols)

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
    print("SLICE-AWARE FEATURE EXTENSION: QF V3 + SERVE V2")
    print("=" * 70)
    print("Nowe cechy: seed context, sila przeciwnikow przed QF, tournament pressure,")
    print("oraz serwis warunkowy pod nawierzchnie, presje, mocnych rywali i matchup reki.")
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


run_sliceaware_qfserve_v3()