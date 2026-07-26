"""Structural benchmark for the 16 rewrites on this base.

Answers Q1 ("do they help?") structurally: for each rewrite, build a pipeline that
triggers it and count DAG ops with the rewrite OFF vs ON (only that rewrite), plus the
combined all-off vs all-on. No dataset needed.

Run:  PYTHONPATH=<repo> .venv/bin/python benchmarks/rewrites/structural_bench.py
"""
import numpy as np
import pandas as pd
import stratum as st
from stratum.optimizer._optimize import optimize, OptConfig
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
import dataclasses

FLAGS = [f.name for f in dataclasses.fields(AlgebraicRewritesConfig)]
ALL_OFF = {k: False for k in FLAGS}

_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})

# (label, flag, builder, needs_cse, needs_folding)
CASES = [
    ("x*1 -> x",            "identity_op",        lambda: st.as_data_op(7) * 1,                                   False, False),
    ("x*0 -> 0",            "any_mul_zero",       lambda: st.as_data_op(7) * 0,                                   False, False),
    ("x+0 -> x",            "add_zero",           lambda: st.as_data_op(7) + 0,                                   False, False),
    ("x-0 -> x",            "identity_subtract",  lambda: st.as_data_op(7) - 0,                                   False, False),
    ("x/1 -> x",            "div_by_one",         lambda: st.as_data_op(7) / 1,                                   False, False),
    ("x**1 -> x",           "pow_by_one",         lambda: st.as_data_op(7) ** 1,                                  False, False),
    ("x**0 -> 1",           "pow_zero",           lambda: st.as_data_op(7) ** 0,                                  False, False),
    ("neg(neg(x)) -> x",    "neg_neg",            lambda: st.as_data_op(7).skb.apply_func(np.negative).skb.apply_func(np.negative), False, False),
    ("abs(abs(x)) -> abs",  "abs_abs",            lambda: st.as_data_op(-7).skb.apply_func(np.abs).skb.apply_func(np.abs), False, False),
    ("exp(x)-1 -> expm1",   "exp_minus_one",      lambda: st.as_data_op(2).skb.apply_func(np.exp) - 1,            False, False),
    ("log(x+1) -> log1p",   "log_plus_one",       lambda: (st.as_data_op(3) + 1).skb.apply_func(np.log),         False, False),
    ("log(sum(exp)) -> lse", "log_sum_exp",       lambda: st.as_data_op(np.array([1.,2.,3.])).skb.apply_func(np.exp).skb.apply_func(np.sum).skb.apply_func(np.log), False, False),
    ("exp/sum(exp)->softmax","softmax",           lambda: (lambda e: e / e.skb.apply_func(np.sum))(st.as_data_op(np.array([1.,2.,3.])).skb.apply_func(np.exp)), True, False),
    ("const fold log(1)->0","constant_folding",   lambda: st.as_data_op(1).skb.apply_func(np.log),                False, True),
    ("select c1[c2] fuse",  "consecutive_select", lambda: st.var("d", _df)[["a", "b", "c"]][["a", "b"]],          False, False),
    ("drop;drop fuse",      "consecutive_drop",   lambda: st.var("d", _df).drop(columns=["a"]).drop(columns=["b"]), False, False),
]


def n_ops(builder, arc, cse=True):
    from stratum._config import FLAGS as F
    old = F.cse
    F.cse = cse
    try:
        env = builder().skb.get_data()  # resolve variables -> frame types known (as evaluate() does)
        out, *_ = optimize(builder(), config=OptConfig(algebraic_rewrites=(arc is not None),
                                                       algebraic_rewrite_config=arc or AlgebraicRewritesConfig(**ALL_OFF)),
                           env=env)
        return len(out)
    finally:
        F.cse = old


def main():
    print(f"{'rewrite':24} {'ops off':>8} {'ops on':>7} {'Δ':>4}  fires")
    print("-" * 55)
    total_off = total_on = 0
    for label, flag, build, needs_cse, needs_fold in CASES:
        only = dict(ALL_OFF); only[flag] = True
        try:
            off = n_ops(build, None, cse=needs_cse)
            on = n_ops(build, AlgebraicRewritesConfig(**only), cse=needs_cse)
            fires = "yes" if on < off else "NO"
            print(f"{label:24} {off:>8} {on:>7} {off-on:>4}  {fires}")
            total_off += off; total_on += on
        except Exception as e:
            print(f"{label:24} {'ERR':>8}  {type(e).__name__}: {e}")
    print("-" * 55)
    print(f"{'TOTAL (isolated)':24} {total_off:>8} {total_on:>7} {total_off-total_on:>4}")


if __name__ == "__main__":
    main()
