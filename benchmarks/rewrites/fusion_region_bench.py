"""Does CSE shrink fusable regions? (the meta-fold's key open question)

The meta-fold fuses a *maximal connected region* of elementwise ops, but it may
only absorb an op whose value is internal to the region -- an op consumed from
outside must stay materialized, or fusing would force recomputation.

CSE deduplicates shared sub-expressions into single nodes with fan-out >= 2.
On a batch of near-duplicate pipelines -- Stratum's actual workload -- that is
exactly the shape that *blocks* region growth. So the two optimizations may work
against each other, and the effect only exists at batch scale.

This harness measures, as the batch size N grows:
  - how many maximal fusable regions exist
  - their mean/max size (ops per region)
  - how many ops sit in a region of size >= 2 (i.e. are actually fusable)
  - how many distinct region *shapes* there are (the kernel-cache hit rate:
    identical shapes compile once and are reused across pipelines)

A negative result is still a result: if CSE systematically shrinks regions, the
Rust kernel should be designed for shared intermediates rather than long chains.

Run:  PYTHONPATH=<repo> .venv/bin/python benchmarks/rewrites/fusion_region_bench.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stratum as st
from stratum._config import FLAGS
from stratum.optimizer._optimize import optimize
from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType

# Elementwise types the meta-fold can fuse. GENERIC is excluded because it may
# wrap a reduction (np.sum) or a stability fusion (scipy softmax); SUM is a
# reduction and is a hard boundary. Mirrors FUSABLE_* in the design plan.
FUSABLE = {
    NumericOpType.ADD, NumericOpType.SUBTRACT, NumericOpType.MULTIPLY,
    NumericOpType.DIVIDE, NumericOpType.POW, NumericOpType.LOG,
    NumericOpType.EXP, NumericOpType.SQRT, NumericOpType.ABS,
    NumericOpType.SQUARE, NumericOpType.LOG1P, NumericOpType.EXPM1,
    NumericOpType.NEGATIVE,
}


def is_fusable(op) -> bool:
    return (isinstance(op, NumericOp) and op.type in FUSABLE
            and not op.args and not op.kwargs)


def find_regions(dag):
    """Group fusable ops into maximal regions.

    An op joins its producer's region only if that producer is fusable *and* has
    exactly one consumer -- the fan-out rule. An op with 2+ consumers ends the
    region, because its value is needed elsewhere and must stay materialized.
    """
    region_of = {}
    regions = []
    for op in dag:
        if not is_fusable(op):
            continue
        joined = None
        for producer in op.inputs:
            if (is_fusable(producer) and len(producer.outputs) == 1
                    and id(producer) in region_of):
                joined = region_of[id(producer)]
                break
        if joined is None:
            joined = []
            regions.append(joined)
        joined.append(op)
        region_of[id(op)] = joined
    return regions


def region_shape(region) -> tuple:
    """Structural key for a region -- what a compiled-kernel cache would key on."""
    return tuple(sorted(op.type.value for op in region))


def build_batch(df, n_variants):
    x = st.var("d", df)
    shared = ((x["a"] - 1.0) / 2.0).skb.apply_func(np.log)
    variants = []
    for i in range(n_variants):
        t = ((x["a"] - 1.0) / 2.0).skb.apply_func(np.log)
        t = t.skb.apply_func(np.exp) * 2.0 - 1.0
        t = t.skb.apply_func(np.abs) + float(i)
        variants.append(t + shared)
    return st.choose_from(variants, name="pipeline").as_data_op()


def measure(df, n_variants, cse):
    old = FLAGS.cse
    FLAGS.cse = cse
    try:
        dag, *_ = optimize(build_batch(df, n_variants), env={"d": df})
    finally:
        FLAGS.cse = old
    regions = [r for r in find_regions(dag) if len(r) >= 2]
    sizes = [len(r) for r in regions] or [0]
    shapes = {region_shape(r) for r in regions}
    fusable_ops = sum(sizes)
    return {
        "ops": len(dag),
        "regions": len(regions),
        "mean_size": float(np.mean(sizes)),
        "max_size": max(sizes),
        "fusable_ops": fusable_ops,
        "shapes": len(shapes),
    }


def main():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.random(50_000) + 1.0,
                       "b": rng.random(50_000) + 1.0})

    print("Fusable-region structure vs batch size")
    print("Question: does CSE (which creates fan-out on shared nodes) shrink the")
    print("regions the meta-fold can fuse?\n")

    for cse in (False, True):
        print(f"  CSE {'ON ' if cse else 'OFF'}")
        print(f"    {'N':>3}{'ops':>7}{'regions':>9}{'mean sz':>9}"
              f"{'max sz':>8}{'fusable ops':>13}{'distinct shapes':>17}")
        for n in (1, 2, 4, 8):
            m = measure(df, n, cse)
            print(f"    {n:>3}{m['ops']:>7}{m['regions']:>9}{m['mean_size']:>9.2f}"
                  f"{m['max_size']:>8}{m['fusable_ops']:>13}{m['shapes']:>17}")
        print()

    print("Reading the table:")
    print("  - 'distinct shapes' << 'regions' means a compiled-kernel cache pays off:")
    print("    the same kernel is reused across pipelines in the batch.")
    print("  - if 'mean sz' drops as N grows with CSE on, CSE is fragmenting the")
    print("    regions and the two optimizations partly work against each other.")


if __name__ == "__main__":
    main()
