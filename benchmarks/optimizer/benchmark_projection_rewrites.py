"""Benchmark for the consecutive select/drop rewrites in
   stratum/optimizer/_projection_rewrites.py
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import stratum as st

from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.ir._dataframe_ops import DropOp
from stratum.optimizer.ir._ops import GetItemOp
from stratum.runtime._scheduler import SequentialScheduler


def make_wide_frame(n_rows: int, n_cols: int, seed: int = 42) -> pd.DataFrame:
    # One float64 matrix -> pandas stores it as a single block, which is the
    # worst case for select/drop cost and so the best case to see the rewrite pay off.
    rng = np.random.default_rng(seed)
    cols = [f"c{i}" for i in range(n_cols)]
    return pd.DataFrame(rng.random((n_rows, n_cols)), columns=cols)


def _run_and_time(dag, rewrite_enabled: bool, fused_op_type) -> tuple[int, float]:
    # Toggle the rewrite: default config has both rewrites on, off config
    # disables them so we can compare before and after.
    cfg = (
        OptConfig()
        if rewrite_enabled
        else OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(
            consecutive_select=False, consecutive_drop=False))
    )
    linearized_dag, split_pos, flagged_ops = optimize(dag, cfg, env=dag.skb.get_data())
    n_ops = sum(1 for op in linearized_dag if isinstance(op, fused_op_type))

    sched = SequentialScheduler(linearized_dag, split_pos, flagged_ops)
    t0 = time.perf_counter()
    try:
        sched.compute_xy()
    except RuntimeError as e:
        if "X and y nodes not found in the DAG" not in str(e):
            raise 
    exec_time = time.perf_counter() - t0  # only execution time
    return n_ops, exec_time


def benchmark_select_rewrite(n_rows: int, n_cols: int, repeats: int = 3) -> None:
    df = make_wide_frame(n_rows, n_cols)
    cols = list(df.columns)
    cols1 = cols[: int(n_cols * 0.75)]           # first select: keep 75% of columns
    cols2 = cols1[: max(1, int(n_cols * 0.15))]  # second select: narrow down hard

    print(f"\n[select rewrite] rows={n_rows:,} cols={n_cols:,} "
          f"({len(cols1)} -> {len(cols2)} columns kept)")
    for rewrite_enabled in (False, True):  # run before first, then after
        times = []
        for _ in range(repeats):
            dag = st.var("d", df)[cols1][cols2]
            n_ops, dt = _run_and_time(dag, rewrite_enabled, GetItemOp)
            times.append(dt)
        avg = sum(times) / len(times)
        label = "after rewrite  " if rewrite_enabled else "before rewrite"
        print(f"  {label} (select ops left={n_ops}): avg={avg:.4f}s over {repeats} runs "
              f"(min={min(times):.4f}s)")


def benchmark_drop_rewrite(n_rows: int, n_cols: int, repeats: int = 3) -> None:
    df = make_wide_frame(n_rows, n_cols)
    cols = list(df.columns)
    drop1 = cols[: int(n_cols * 0.25)]                     # first drop: remove 25%
    drop2 = cols[int(n_cols * 0.25): int(n_cols * 0.375)]  # second drop: remove another 12.5%

    print(f"\n[drop rewrite] rows={n_rows:,} cols={n_cols:,} "
          f"(drop {len(drop1)} then {len(drop2)} more columns)")
    for rewrite_enabled in (False, True):
        times = []
        for _ in range(repeats):
            dag = st.var("d", df).drop(columns=drop1).drop(columns=drop2)
            n_ops, dt = _run_and_time(dag, rewrite_enabled, DropOp)
            times.append(dt)
        avg = sum(times) / len(times)
        label = "after rewrite  " if rewrite_enabled else "before rewrite"
        print(f"  {label} (drop ops left={n_ops}):   avg={avg:.4f}s over {repeats} runs "
              f"(min={min(times):.4f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=200_000)
    parser.add_argument("--n-cols", type=int, nargs="+", default=[100, 400, 1000], help="one or more column counts to sweep over")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    for n_cols in args.n_cols:
        benchmark_select_rewrite(args.n_rows, n_cols, args.repeats)
        benchmark_drop_rewrite(args.n_rows, n_cols, args.repeats)


if __name__ == "__main__":
    main()
