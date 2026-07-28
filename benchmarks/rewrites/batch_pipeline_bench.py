"""Batch benchmark: rewrites on Stratum's actual workload (a batch of pipelines).

Why this exists. The other harnesses measure one pipeline at a time, which is the
setting Stratum is *not* built for. The paper (arXiv:2603.03589) fuses a *batch*
of agent-generated pipelines into one DAG, and attributes the logical layer's
2.2x mainly to CSE deduplicating shared preprocessing. Measuring our rewrites
without CSE present therefore answers a question nobody asked.

Workload mirrors the paper's evaluation: iteration 1 explores
2 preprocessing strategies x 4 models = 8 pipelines that share preprocessing.
We express that as a Stratum `choose_from`, which `choice_unrolling` expands into
one DAG -- the same structure the paper describes.

Factors (2x2):  CSE {off, on} x rewrites {off, on}
so the rewrites' contribution can be read *on top of* CSE rather than instead of
it, which is the honest attribution.

Reports per cell: DAG op count, wall-clock (median), and peak RSS measured in a
subprocess (same technique as memory_bench.py).

Run:  PYTHONPATH=<repo> .venv/bin/python benchmarks/rewrites/batch_pipeline_bench.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import stratum as st
from stratum._config import FLAGS
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.optimizer._optimize import OptConfig, optimize

ALL_FLAGS = [f.name for f in dataclasses.fields(AlgebraicRewritesConfig)]
ALL_OFF = {k: False for k in ALL_FLAGS}

N_ROWS = int(os.environ.get("BENCH_ROWS", "200000"))
REPEATS = 5


def make_frame(n_rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.random(n_rows) + 1.0,
        "b": rng.random(n_rows) + 1.0,
        "c": rng.random(n_rows) + 1.0,
    })


# Redundancy patterns an agent might emit. Each is a (label, builder) pair applied
# to a variant's tail. Keeping them distinct is what separates CSE's contribution
# from the rewrites': identical redundancy across variants is deduplicated by CSE
# *before* the rewrites see it, so the rewrites appear to contribute little.
REDUNDANCY_PATTERNS = [
    ("mul1_add0",  lambda t: t * 1 + 0),
    ("pow1",       lambda t: t ** 1),
    ("negneg",     lambda t: t.skb.apply_func(np.negative).skb.apply_func(np.negative)),
    ("div1",       lambda t: t / 1),
    ("sub0",       lambda t: t - 0),
    ("mul1_pow1",  lambda t: (t * 1) ** 1),
    ("add0_div1",  lambda t: (t + 0) / 1),
    ("negneg_sub0", lambda t: t.skb.apply_func(np.negative)
                               .skb.apply_func(np.negative) - 0),
]


def build_batch(df: pd.DataFrame, n_variants: int = 8, uniform: bool = False):
    """A batch of `n_variants` near-duplicate pipelines over shared preprocessing.

    Structure follows the paper's iteration 1: a shared feature-engineering prefix
    (the redundancy CSE is meant to exploit) and per-variant tails that differ
    slightly -- the "median iteration changes 16% or fewer lines" pattern.

    The tails carry deliberately redundant algebra because that is what the
    rewrites eliminate; a batch without redundancy would measure nothing.

    `uniform` selects *how* that redundancy is distributed, which turns out to
    determine what the benchmark actually measures:

    - ``uniform=True``  -- every variant carries the *same* redundant algebra.
      CSE then deduplicates it across variants before the rewrites run, so the
      rewrites' measured contribution is small. This is an artifact of the
      construction, not a property of the rewrites.
    - ``uniform=False`` (default) -- each variant carries a *different* pattern,
      so the redundancy is not shared and CSE cannot collapse it. This isolates
      the rewrites' own contribution and is the more realistic model of agent
      output, where successive iterations rewrite different parts of the script.
    """
    x = st.var("d", df)

    # Shared prefix: identical across every variant, so CSE should collapse it.
    # This is the cross-pipeline redundancy the paper attributes CSE's win to.
    shared = (x["a"] - 1.0) / 2.0
    shared = shared.skb.apply_func(np.log)

    variants = []
    for i in range(n_variants):
        # Each variant re-derives the same prefix (agents emit whole scripts,
        # not diffs), then applies its own tail.
        t = (x["a"] - 1.0) / 2.0
        t = t.skb.apply_func(np.log)

        if uniform:
            t = t * 1 + 0
            t = t ** 1
            t = t.skb.apply_func(np.negative).skb.apply_func(np.negative)
            t = t / 1
        else:
            _, pattern = REDUNDANCY_PATTERNS[i % len(REDUNDANCY_PATTERNS)]
            t = pattern(t)

        t = t + float(i)                  # the "16% of lines" difference
        variants.append(t + shared)

    # `.as_data_op()` turns the Choice back into a DataOp so it can be optimized;
    # `choice_unrolling` then expands it into one DAG per variant.
    return st.choose_from(variants, name="pipeline").as_data_op()


def n_ops_and_time(df, cse: bool, rewrites: bool, uniform: bool = False,
                   repeats: int = REPEATS):
    old_cse = FLAGS.cse
    FLAGS.cse = cse
    try:
        arc = (AlgebraicRewritesConfig() if rewrites
               else AlgebraicRewritesConfig(**ALL_OFF))
        cfg = OptConfig(algebraic_rewrites=True, algebraic_rewrite_config=arc)

        dag, *_ = optimize(build_batch(df, uniform=uniform), config=cfg,
                           env={"d": df})
        n_ops = len(dag)

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            optimize(build_batch(df, uniform=uniform), config=cfg, env={"d": df})
            times.append((time.perf_counter() - t0) * 1e3)
        return n_ops, float(np.median(times))
    finally:
        FLAGS.cse = old_cse


_CHILD = r"""
import json, os, resource, sys
sys.path.insert(0, os.environ["BENCH_REPO"])
import numpy as np, pandas as pd, dataclasses
import stratum as st
from stratum._config import FLAGS
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.optimizer._optimize import OptConfig, optimize
sys.path.insert(0, os.path.join(os.environ["BENCH_REPO"], "benchmarks", "rewrites"))
from batch_pipeline_bench import build_batch, make_frame, ALL_OFF

cse = os.environ["BENCH_CSE"] == "1"
rw  = os.environ["BENCH_RW"] == "1"
df  = make_frame(int(os.environ["BENCH_ROWS"]))
FLAGS.cse = cse
arc = AlgebraicRewritesConfig() if rw else AlgebraicRewritesConfig(**ALL_OFF)
uni = os.environ.get("BENCH_UNIFORM") == "1"
dag, *_ = optimize(build_batch(df, uniform=uni),
                   config=OptConfig(algebraic_rewrites=True,
                                    algebraic_rewrite_config=arc),
                   env={"d": df})
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"ops": len(dag), "peak_bytes": peak}))
"""


def peak_rss(repo: str, cse: bool, rewrites: bool, uniform: bool = False) -> int:
    """Peak RSS of a fresh interpreter doing one optimize() of the batch."""
    env = dict(os.environ, BENCH_REPO=repo, BENCH_CSE="1" if cse else "0",
               BENCH_RW="1" if rewrites else "0", BENCH_ROWS=str(N_ROWS),
               BENCH_UNIFORM="1" if uniform else "0")
    out = subprocess.run([sys.executable, "-c", _CHILD], env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        return -1
    return json.loads(out.stdout.strip().splitlines()[-1])["peak_bytes"]


def run_grid(df, repo, uniform, label):
    print(f"\n{label}")
    print(f"  {'CSE':<6}{'rewrites':<10}{'ops':>6}{'optimize ms':>14}"
          f"{'peak RSS MB':>14}")
    cells = {}
    for cse in (False, True):
        for rw in (False, True):
            ops, ms = n_ops_and_time(df, cse, rw, uniform=uniform)
            rss = peak_rss(repo, cse, rw, uniform)
            cells[(cse, rw)] = (ops, ms, rss)
            rss_s = f"{rss / 1e6:>13.1f}" if rss > 0 else f"{'n/a':>13}"
            print(f"  {'on' if cse else 'off':<6}{'on' if rw else 'off':<10}"
                  f"{ops:>6}{ms:>14.2f}{rss_s}")

    o_off = cells[(False, False)][0]
    o_cse = cells[(True, False)][0]
    o_rw = cells[(False, True)][0]
    o_both = cells[(True, True)][0]
    print(f"    CSE alone             : {o_off:>3} -> {o_cse:>3} ops "
          f"({o_off - o_cse:>2} removed)")
    print(f"    rewrites alone        : {o_off:>3} -> {o_rw:>3} ops "
          f"({o_off - o_rw:>2} removed)")
    print(f"    rewrites on top of CSE: {o_cse:>3} -> {o_both:>3} ops "
          f"({o_cse - o_both:>2} removed)   <-- our contribution")
    return o_cse - o_both


def main():
    repo = os.environ.get("BENCH_REPO", os.getcwd())
    df = make_frame(N_ROWS)
    print(f"Batch pipeline benchmark -- 8 variants over a shared prefix, "
          f"{N_ROWS:,} rows")
    print("Objective (paper 4.3): minimize execution time under memory "
          "constraints,\nso both wall-clock and peak RSS are reported.")
    print("\nTwo constructions of the same batch, differing only in how the")
    print("redundant algebra is distributed across the eight variants.")

    d_uniform = run_grid(df, repo, True,
                         "[A] uniform redundancy -- every variant carries the "
                         "SAME redundant algebra")
    d_varied = run_grid(df, repo, False,
                        "[B] varied redundancy -- each variant carries a "
                        "DIFFERENT pattern")

    print("\nWhy the two differ")
    print(f"  rewrites' contribution on top of CSE:  [A] {d_uniform} ops   "
          f"[B] {d_varied} ops")
    print("  In [A] the redundancy is identical across variants, so CSE collapses")
    print("  it to one copy before the rewrites run -- they then have only that")
    print("  copy left to eliminate. In [B] the redundancy differs per variant, so")
    print("  CSE cannot dedup it and each pattern must be rewritten on its own.")
    print("  [B] is the more realistic model of agent output and isolates the")
    print("  rewrites' own contribution; [A] understates it by construction.")


if __name__ == "__main__":
    main()
