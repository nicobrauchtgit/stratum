"""Micro-benchmark: does single-pass fused execution beat per-operator NumPy?

Motivation. The 16 rewrites reduce *operator count*, but each surviving NumericOp
still runs as an independent NumPy call: one full pass over the column plus a
fresh intermediate allocation. This harness measures what a *fused kernel* buys
instead -- one pass, no intermediates -- using numexpr as the executor.

It deliberately measures the executor in isolation (no Stratum DAG), so the
numbers bound what the meta-fold can deliver before any optimizer overhead.

Reports, per expression and input size:
  - numpy chain (status quo) vs numexpr fused, median ms
  - speedup, and the size at which fusion becomes profitable (the cost gate)
  - peak memory (tracemalloc) as a multiple of the input array

Run:  PYTHONPATH=<repo> .venv/bin/python benchmarks/rewrites/fusion_micro_bench.py
"""
from __future__ import annotations

import time
import tracemalloc

import numexpr as ne
import numpy as np

# (label, numpy chain, equivalent numexpr expression, #elementwise ops)
CASES = [
    ("standardize   (x-m)/s",
     lambda x: np.divide(np.subtract(x, 2.0), 3.0),
     "(x - 2.0) / 3.0", 2),
    ("log-chain     log((x-2)/3)*2",
     lambda x: np.multiply(np.log(np.divide(np.subtract(x, 2.0), 3.0)), 2.0),
     "log((x - 2.0) / 3.0) * 2.0", 4),
    ("poly3         x^3+2x^2+3x+4",
     lambda x: np.add(np.add(np.add(np.multiply(np.multiply(x, x), x),
                                    np.multiply(np.square(x), 2.0)),
                             np.multiply(x, 3.0)), 4.0),
     "x*x*x + x**2 * 2.0 + x * 3.0 + 4.0", 8),
    ("logabs        log(abs(x)+1)*2",
     lambda x: np.multiply(np.log(np.add(np.abs(x), 1.0)), 2.0),
     "log(abs(x) + 1.0) * 2.0", 4),
]

SIZES = [1_000, 10_000, 50_000, 100_000, 1_000_000, 10_000_000]
REPEATS = 7
WARMUP = 3


def _median_ms(fn, repeats=REPEATS):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def _peak_bytes(fn):
    fn()  # warm caches so we measure the computation, not first-touch
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main():
    rng = np.random.default_rng(0)
    print("Fused elementwise execution: numpy chain vs numexpr single pass")
    print("(float64; median of %d runs after %d warm-ups)\n" % (REPEATS, WARMUP))

    crossovers = {}
    for label, np_fn, expr, n_ops in CASES:
        print(f"{label}   [{n_ops} elementwise ops]   expr: {expr}")
        print(f"  {'n':>10}  {'numpy ms':>10}  {'numexpr ms':>11}  {'speedup':>8}")
        crossover = None
        for n in SIZES:
            x = rng.random(n) + 1.0
            with np.errstate(all="ignore"):
                a = np_fn(x)
                b = ne.evaluate(expr, local_dict={"x": x})
                if not np.allclose(a, b, equal_nan=True):
                    print(f"  {n:>10}  MISMATCH -- results differ, skipping")
                    continue
                t_np = _median_ms(lambda: np_fn(x))
                t_ne = _median_ms(lambda: ne.evaluate(expr, local_dict={"x": x}))
            speed = t_np / t_ne
            if crossover is None and speed >= 1.0:
                crossover = n
            print(f"  {n:>10}  {t_np:>10.3f}  {t_ne:>11.3f}  {speed:>7.2f}x")
        crossovers[label] = crossover
        print(f"  -> profitable from n >= {crossover}\n"
              if crossover else "  -> never profitable at these sizes\n")

    # Peak memory at a size where fusion is clearly profitable.
    n = 5_000_000
    x = np.random.default_rng(0).random(n) + 1.0
    print(f"Peak memory at n={n:,} (array itself = {x.nbytes / 1e6:.1f} MB)")
    print(f"  {'case':<32}{'numpy':>12}{'numexpr':>12}{'ratio':>10}")
    for label, np_fn, expr, _ in CASES:
        with np.errstate(all="ignore"):
            p_np = _peak_bytes(lambda: np_fn(x))
            p_ne = _peak_bytes(lambda: ne.evaluate(expr, local_dict={"x": x}))
        print(f"  {label.split()[0]:<32}{p_np / x.nbytes:>11.2f}x{p_ne / x.nbytes:>11.2f}x"
              f"{p_np / max(p_ne, 1):>9.2f}x")

    print("\nCost gate: fusion must be skipped below the crossover, or it is a "
          "regression.\n  crossovers: "
          + ", ".join(f"{k.split()[0]}={v}" for k, v in crossovers.items()))


if __name__ == "__main__":
    main()
