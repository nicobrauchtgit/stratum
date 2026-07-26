"""Phase-ordering benchmark: does the order rewrites are applied (= the order they were
merged into algebraic_rewrites()) change the optimized result?

Each pass runs ONCE in a fixed sequence. When rewrite A *enables* rewrite B (A removes a
node that was blocking B's pattern), B only fires if it runs AFTER A. So the merge/dispatch
order is a phase-ordering problem (cf. Stratum paper: "Rewrite ordering is workload-dependent").

Run:  PYTHONPATH=<repo> .venv/bin/python benchmarks/rewrites/order_bench.py
"""
import itertools
import numpy as np
import pandas as pd
import stratum as st
from stratum._config import FLAGS
from stratum.optimizer._optimize import (
    convert_to_ops, add_splitting_op, extract_frame_operators,
    extract_numeric_operators, run_op_cse_pass,
)
from stratum.optimizer._linearization import linearize_dag
from stratum.optimizer import _numeric_rewrites as NR

P = {
    "identity_op": NR.eliminate_identity_operation, "add_zero": NR.eliminate_add_zero,
    "div_by_one": NR.eliminate_div_by_one, "abs_abs": NR.eliminate_abs_abs,
    "neg_neg": NR.eliminate_neg_neg, "log_exp": NR.eliminate_log_exp,
    "any_mul_zero": NR.eliminate_any_mul_zero, "pow_by_one": NR.eliminate_pow_by_one,
    "constant_folding": NR.eliminate_constant_folding,
}
_s = pd.Series([1.0, -2.0, 3.0])


def ir(builder):
    r = convert_to_ops(builder(), None)
    r = add_splitting_op(r)
    r = extract_frame_operators(r)
    r = extract_numeric_operators(r)
    if FLAGS.cse:
        r = run_op_cse_pass(r)
    return r


def n_after(builder, order):
    r = ir(builder)
    for name in order:
        r = P[name](r)
    return len(linearize_dag(r)[0])


def fixpoint(builder, names, max_iter=10):
    r = ir(builder); prev = None
    for _ in range(max_iter):
        for name in names:
            r = P[name](r)
        cur = len(linearize_dag(r)[0])
        if cur == prev:
            return cur
        prev = cur
    return cur


def study(label, builder, names):
    perms = list(itertools.permutations(names))
    res = {o: n_after(builder, o) for o in perms}
    counts = sorted(set(res.values()))
    fp = fixpoint(builder, names)
    sens = len(counts) > 1
    print(f"\n### {label}")
    print(f"    passes {names}: distinct final op-counts across {len(perms)} orders = {counts} | fixpoint = {fp}")
    print(f"    ORDER-SENSITIVE: {'YES' if sens else 'no'}")
    if sens:
        best = min(res.values()); worst = max(res.values())
        ob = next(o for o, v in res.items() if v == best); ow = next(o for o, v in res.items() if v == worst)
        print(f"      best  {best} ops: {' -> '.join(ob)}")
        print(f"      worst {worst} ops: {' -> '.join(ow)}  (missed rewrite: an enabling pass ran too late)")


def main():
    print("=" * 72)
    print("PHASE-ORDERING STUDY (final DAG op-count vs pass order)")
    print("=" * 72)

    # ---- Enabling interactions: A removes a node blocking B's pattern ----
    study("abs( abs(x) * 1 )   [identity_op enables abs_abs]",
          lambda: st.var("x", _s).skb.apply_func(np.abs).__mul__(1).skb.apply_func(np.abs),
          ["identity_op", "abs_abs"])

    study("neg( neg(x) + 0 )   [add_zero enables neg_neg]",
          lambda: st.var("x", _s).skb.apply_func(np.negative).__add__(0).skb.apply_func(np.negative),
          ["add_zero", "neg_neg"])

    study("abs( abs(x*1) * 1 ) [identity enables abs_abs, deeper]",
          lambda: st.var("x", _s).__mul__(1).skb.apply_func(np.abs).__mul__(1).skb.apply_func(np.abs),
          ["identity_op", "abs_abs"])

    # ---- Confluent contrast: disjoint patterns, order-independent ----
    study("(x/1)**1 then neg(neg)  [disjoint patterns]",
          lambda: (st.var("x", _s).__truediv__(1).__pow__(1)).skb.apply_func(np.negative).skb.apply_func(np.negative),
          ["div_by_one", "pow_by_one", "neg_neg"])


if __name__ == "__main__":
    main()
