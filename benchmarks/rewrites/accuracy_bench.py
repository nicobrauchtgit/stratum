"""Numerical-accuracy benchmark for the fusion/stability rewrites.

Shows the win the wall-clock benchmark cannot: fused ops (log1p, expm1, logsumexp,
softmax) are numerically stable where the naive chain loses precision or overflows.
Runs the pipeline through the optimizer with rewrites OFF (naive chain) vs ON (fused op)
and compares both to a high-precision reference.

Run:  PYTHONPATH=<repo> .venv/bin/python benchmarks/rewrites/accuracy_bench.py
"""
import math
import numpy as np
import scipy.special as sp
import stratum as st
from stratum.optimizer._optimize import optimize, OptConfig
from stratum.runtime._scheduler import SequentialScheduler


def run(builder, on):
    dag = builder(); env = dag.skb.get_data()
    lin, sp_, fl = optimize(dag, config=OptConfig(algebraic_rewrites=on), env=env)
    sched = SequentialScheduler(lin, sp_, fl, False)
    sched.mode = "fit_transform"
    for node in lin:                      # inline compute() but keep the final result
        sched.process_op(node)
    return sched.pool.pin(lin[-1])


def rel_err(got, ref):
    got = np.asarray(got, float); ref = np.asarray(ref, float)
    denom = np.where(np.abs(ref) > 0, np.abs(ref), 1.0)
    return float(np.max(np.abs(got - ref) / denom))


def case(label, builder, ref):
    naive = np.asarray(run(builder, False), float)
    fused = np.asarray(run(builder, True), float)
    ref = np.asarray(ref, float)
    print(f"\n### {label}")
    print(f"    naive (rewrites off): {naive}   rel-err {rel_err(naive, ref):.3e}"
          f"{'   <-- OVERFLOW/NaN' if not np.all(np.isfinite(naive)) else ''}")
    print(f"    fused (rewrites on):  {fused}   rel-err {rel_err(fused, ref):.3e}")
    print(f"    reference:            {ref}")


def main():
    print("=" * 70)
    print("NUMERICAL ACCURACY: naive chain vs fused/stable op")
    print("=" * 70)

    # log(1+x) for tiny x: catastrophic cancellation in the naive form.
    x = 1e-12
    case("log(1+x), x=1e-12  [log1p]",
         lambda: (st.as_data_op(x) + 1).skb.apply_func(np.log),
         [math.log1p(x)])

    # exp(x)-1 for tiny x.
    case("exp(x)-1, x=1e-12  [expm1]",
         lambda: st.as_data_op(x).skb.apply_func(np.exp) - 1,
         [math.expm1(x)])

    # softmax with large inputs: naive exp overflows to inf -> NaN.
    big = np.array([1000.0, 1001.0, 1002.0])
    e = lambda: st.as_data_op(big).skb.apply_func(np.exp)
    case("softmax([1000,1001,1002])  [stable softmax]",
         lambda: e() / e().skb.apply_func(np.sum),
         sp.softmax(big))

    # logsumexp with large inputs: naive sum(exp) overflows.
    case("log(sum(exp([1000,1001,1002])))  [logsumexp]",
         lambda: st.as_data_op(big).skb.apply_func(np.exp).skb.apply_func(np.sum).skb.apply_func(np.log),
         [sp.logsumexp(big)])


if __name__ == "__main__":
    main()
