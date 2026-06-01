"""
Walidacja walk-forward wariantow slice-aware (sliceaware / bestof5 / qfserve)
============================================================================

Warianty slice-aware byly dotad oceniane TYLKO na pojedynczym tescie. Tu robimy
to uczciwie: walk-forward przez kilka sezonow + test parowany McNemar na tych
samych meczach (baseline vs wariant).

Mechanizm: monkey-patch runpy.run_path cache'uje namespace baseline dla biezacego
roku, dzieki czemu baseline liczy sie RAZ na rok, a 3 warianty go reuzywaja
(jak w slicecompare). Per rok zbieramy winner_perspective baseline i kazdego
wariantu (match_id + correct_prediction), parujemy po match_id, liczymy delte
i McNemar pooled po wszystkich latach.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

WORKDIR = Path(__file__).resolve().parent
BASELINE_SCRIPT_PATH = (WORKDIR / "main_48_cech.py").resolve()

_wf_env = os.environ.get("TENNIS_WF_YEARS")
TARGET_YEARS = ([int(y) for y in _wf_env.split(",")] if _wf_env
                else [2022, 2023, 2024, 2025])

VARIANTS = {
    "sliceaware": "main_48_cech_sliceaware.py",
    "bestof5_v1": "main_48_cech_sliceaware_bestof5_v1.py",
    "qfserve_v3": "main_48_cech_sliceaware_qfserve_v3.py",
}

# --- cache baseline per rok (reset na poczatku kazdego roku) ---
_baseline_cache: dict | None = None
_original_run_path = runpy.run_path


def _cached_run_path(path_or_name, *args, **kwargs):
    global _baseline_cache
    try:
        resolved = Path(path_or_name).resolve()
    except (TypeError, OSError):
        resolved = None
    if resolved == BASELINE_SCRIPT_PATH:
        if _baseline_cache is None:
            _baseline_cache = _original_run_path(path_or_name, *args, **kwargs)
        return _baseline_cache
    return _original_run_path(path_or_name, *args, **kwargs)


runpy.run_path = _cached_run_path


def reset_baseline_cache():
    global _baseline_cache
    _baseline_cache = None


def execute_script(script_name: str) -> dict:
    script_path = WORKDIR / script_name
    captured = io.StringIO()
    original_cwd = os.getcwd()
    os.chdir(WORKDIR)
    try:
        with contextlib.redirect_stdout(captured):
            return runpy.run_path(str(script_path))
    finally:
        os.chdir(original_cwd)


def eval_frame(ns) -> pd.DataFrame:
    return ns["winner_perspective"][["match_id", "correct_prediction"]].copy()


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    z = (abs(b - c) - 1) / math.sqrt(n)
    return z, math.erfc(abs(z) / math.sqrt(2))


def main():
    # akumulatory par per wariant
    pairs = {name: [] for name in VARIANTS}  # (base_correct, var_correct)
    per_year = {name: [] for name in VARIANTS}

    for year in TARGET_YEARS:
        print(f"\n===== ROK {year} =====", flush=True)
        os.environ["TENNIS_TARGET_YEAR"] = str(year)
        reset_baseline_cache()

        # baseline raz (zapelnia cache)
        base_ns = execute_script("main_48_cech.py")
        base_eval = eval_frame(base_ns)
        base_match = float(base_ns["match_accuracy"])
        print(f"  baseline match={base_match:.4f} (n={len(base_eval)})", flush=True)

        for name, script in VARIANTS.items():
            var_ns = execute_script(script)  # reuzywa cached baseline
            var_eval = eval_frame(var_ns)
            var_match = float(var_ns["match_accuracy"])
            merged = base_eval.merge(var_eval, on="match_id", suffixes=("_base", "_var"))
            for _, r in merged.iterrows():
                pairs[name].append((bool(r["correct_prediction_base"]), bool(r["correct_prediction_var"])))
            per_year[name].append({"year": year, "baseline": base_match, "variant": var_match,
                                   "delta": var_match - base_match})
            print(f"    {name:<12} match={var_match:.4f}  delta={var_match-base_match:+.4f}", flush=True)

    os.environ.pop("TENNIS_TARGET_YEAR", None)

    print("\n" + "=" * 74)
    print("WALK-FORWARD: warianty slice-aware vs baseline")
    print("=" * 74)
    for name in VARIANTS:
        df = pd.DataFrame(per_year[name])
        arr = np.array(pairs[name])
        base_c, var_c = arr[:, 0], arr[:, 1]
        N = len(arr)
        b = int(np.sum(base_c & ~var_c))
        c = int(np.sum(~base_c & var_c))
        z, p = mcnemar(b, c)
        pooled_delta = var_c.mean() - base_c.mean()
        pos = int((df["delta"] > 0).sum())
        print(f"\n--- {name} ---")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  POOLED ({N} meczow): baseline={base_c.mean():.4f}  {name}={var_c.mean():.4f}  "
              f"delta={pooled_delta:+.4f}  (dodatnie {pos}/{len(df)} lat)")
        print(f"  McNemar: b={b} c={c} z={z:.2f} p={p:.4f} -> "
              f"{'ISTOTNE' if p < 0.05 and c > b else 'brak istotnosci' if p >= 0.05 else 'ISTOTNE na niekorzysc'}")


if __name__ == "__main__":
    main()
