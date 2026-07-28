"""Wall-clock benchmark: (1) rewrites OFF vs ON on different kinds/sizes of data,
and (2) the phase-ordering effect on end-to-end runtime.

Data: synthetic (reproducible) + real tabular columns from OpenML/sklearn
(california-housing 20K, covtype 581K) -- different distributions and sizes. (Kaggle
would need credentials; OpenML/sklearn is the same kind of tabular data without auth.)

Motivation (Stratum paper, Fig. 2): agent-generated pipelines are highly redundant
(median 16% changed lines), so eliminating redundant ops is directly relevant.

Run:  PYTHONPATH=<repo> .venv/bin/python benchmarks/rewrites/walltime_bench.py
"""
import statistics
import time
import numpy as np
import pandas as pd
import stratum as st
from stratum._config import FLAGS
from stratum.optimizer._optimize import (
    OptConfig, optimize, convert_to_ops, add_splitting_op, extract_frame_operators,
    extract_numeric_operators, run_op_cse_pass, compute_pinned_ops, plan_input_removals,
)
from stratum.optimizer._linearization import linearize_dag
from stratum.optimizer import _numeric_rewrites as NR
from stratum.runtime._scheduler import SequentialScheduler

REPEATS = 5
P = {"identity_op": NR.eliminate_identity_operation, "add_zero": NR.eliminate_add_zero,
     "abs_abs": NR.eliminate_abs_abs, "neg_neg": NR.eliminate_neg_neg}


def _median_ms(fn):
    fn()  # warmup
    return statistics.median([(_t(fn)) for _ in range(REPEATS)])


def _t(fn):
    t0 = time.perf_counter(); fn(); return (time.perf_counter() - t0) * 1e3


def run_toggle(builder, on):
    dag = builder(); env = dag.skb.get_data()
    lin, sp, fl = optimize(dag, config=OptConfig(algebraic_rewrites=on), env=env)
    SequentialScheduler(lin, sp, fl, False).compute(0, "fit_transform")


def _ir(builder, env=None):
    r = convert_to_ops(builder(), env); r = add_splitting_op(r)
    r = extract_frame_operators(r); r = extract_numeric_operators(r)
    if FLAGS.cse:
        r = run_op_cse_pass(r)
    return r


def run_order(builder, order, iters=1):
    """Execute the pipeline applying rewrite passes in a chosen order (iters>1 = fixpoint)."""
    root = _ir(builder, env=builder().skb.get_data())  # resolve vars for execution
    for _ in range(iters):
        for name in order:
            root = P[name](root)
    lin, sp, fl = linearize_dag(root)
    pinned = compute_pinned_ops(lin, sp, fl); plan_input_removals(lin, pinned)
    SequentialScheduler(lin, sp, fl, False).compute(0, "fit_transform")
    return len(lin)


def improvement(name, builder):
    off = _median_ms(lambda: run_toggle(builder, False))
    on = _median_ms(lambda: run_toggle(builder, True))
    print(f"    {name:42} off {off:8.2f}ms  on {on:8.2f}ms  speedup {off/on:5.2f}x")


def main():
    # datasets: (label, pandas Series) -- one representative column each
    from sklearn.datasets import fetch_california_housing, fetch_covtype
    rng = np.random.default_rng(0)
    ds = [("synthetic-1M", pd.Series(rng.random(1_000_000) + 0.5))]
    try:
        ch = fetch_california_housing(as_frame=True).frame
        ds.append(("california-MedInc-20K", ch["MedInc"].astype(float).reset_index(drop=True)))
    except Exception as e:
        print("california skipped:", e)
    try:
        cv = fetch_covtype(as_frame=True).frame
        ds.append(("covtype-Elevation-581K", cv["Elevation"].astype(float).reset_index(drop=True)))
    except Exception as e:
        print("covtype skipped:", e)

    print("=" * 78)
    print(f"SECTION 1 — rewrites OFF vs ON (median of {REPEATS})")
    print("=" * 78)
    for label, ser in ds:
        print(f"\n[{label}]  n={len(ser):,}")

        def p_identities(s=ser):
            x = st.var("x", s); x = (x / 1) ** 1; x = x * 1 + 0 - 0
            return x.skb.apply_func(np.negative).skb.apply_func(np.negative)
        improvement("redundant identities (x/1,**1,*1,+0,-0,neg-neg)", p_identities)

        def p_log1p(s=ser):
            return (st.var("x", s.abs()) + 1).skb.apply_func(np.log)
        improvement("log1p fusion (log(x+1))", p_log1p)

        def p_softmax(s=ser):
            e = st.var("x", s).skb.apply_func(np.exp)
            return e / e.skb.apply_func(np.sum)
        improvement("softmax fusion (exp/sum(exp))", p_softmax)

    print("\n" + "=" * 78)
    print(f"SECTION 2 — PHASE ORDERING on end-to-end runtime (median of {REPEATS})")
    print("  pipeline: abs( abs(x)*1 )  -- identity_op must run BEFORE abs_abs")
    print("=" * 78)
    for label, ser in ds:
        col = ser.abs()

        def p_enable(s=col):
            return st.var("x", s).skb.apply_func(np.abs).__mul__(1).skb.apply_func(np.abs)
        good = _median_ms(lambda: run_order(p_enable, ["identity_op", "abs_abs"]))
        bad = _median_ms(lambda: run_order(p_enable, ["abs_abs", "identity_op"]))
        fix = _median_ms(lambda: run_order(p_enable, ["identity_op", "abs_abs"], iters=2))
        n_good = run_order(p_enable, ["identity_op", "abs_abs"])
        n_bad = run_order(p_enable, ["abs_abs", "identity_op"])
        print(f"\n[{label}]  n={len(col):,}")
        print(f"    good order (identity->abs_abs): {good:8.2f}ms  ({n_good} ops)")
        print(f"    bad  order (abs_abs->identity): {bad:8.2f}ms  ({n_bad} ops)   penalty {bad/good:.2f}x")
        print(f"    fixpoint (repeat to converge):  {fix:8.2f}ms  ({n_good} ops)")


if __name__ == "__main__":
    main()
