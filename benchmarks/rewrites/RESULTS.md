# Rewrite Benchmarks — Results (summary)

Full write-up: **`REPORT.md`** (the 8-page report). Base: `nicobrauchtgit/stratum:main`
(16 rewrites). Median of 5 timed runs, warm-up excluded. Hardware: Apple M3 Pro (12c),
36 GB, Python 3.12 / NumPy 2.3 / pandas 3.0 / SciPy 1.16 / sklearn 1.8 / skrub 0.8.

Harnesses (run `PYTHONPATH=<repo> .venv/bin/python benchmarks/rewrites/<script>`):
`structural_bench.py`, `order_bench.py`, `walltime_bench.py`.

## Q1 — do the rewrites help?
- **Structural:** all 16 fire; −19 ops in isolation (deterministic).
- **Wall-clock (OFF→ON):**

  | pipeline | synth 1M | california 20K | covtype 581K |
  |---|---|---|---|
  | redundant identities | 1.23× | 1.04× | **1.35×** |
  | log1p fusion | 0.92× | 1.03× | 0.98× |
  | softmax fusion | 0.93× | 0.98× | 1.01× |

  → Eliminations give small-but-real, scale-dependent speedups on redundant pipelines;
  fusions are speed-neutral (their value is numerical stability, not speed).

## Q2 — does order (of merge/application) matter?  YES, for enabling interactions
- `abs(abs(x)·1)`: `{2,3}` ops across orders (identity_op must precede abs_abs); fixpoint→2.
- `neg(neg(x)+0)`: `{1,3}` ops; fixpoint→1.
- disjoint patterns (`x/1`,`**1`,`neg-neg`): confluent `{1}`.
- Wall-clock penalty of the bad order on `abs(abs(x)·1)`: up to **1.17×** at 581K rows;
  within noise at smaller sizes. Fixpoint iteration removes the dependence at extra cost.

Confirms the Stratum paper's "rewrite ordering is workload-dependent." Also found a missed
optimization: `y*exp(0)` stays `y*1` (identity matches scalar literals, not folded ValueOps).
