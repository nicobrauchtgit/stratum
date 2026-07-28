# Experiments

Eight harnesses in `benchmarks/rewrites/`. Each is a standalone script with no
arguments; captured output for every one is in `benchmarks/rewrites/logs/`.

## Running them

```bash
devbox run install                        # once: nix env, venv, rust extension
export PYTHONPATH=$PWD BENCH_REPO=$PWD
.venv/bin/python benchmarks/rewrites/<script>.py
```

Or all of them, regenerating the logs:

```bash
for b in structural order walltime accuracy memory \
         fusion_micro fusion_region batch_pipeline; do
  .venv/bin/python benchmarks/rewrites/${b}_bench.py \
    > benchmarks/rewrites/logs/${b}_bench.log 2>&1
done
```

`BENCH_REPO` is needed only by `batch_pipeline_bench.py`, which measures peak
RSS in a subprocess. `BENCH_ROWS` (default `200000`) overrides its input size.

**Environment.** Apple M3 Pro (12 cores, 36 GB), macOS arm64; Python 3.12,
NumPy 2.3, pandas 3.0, SciPy 1.16, scikit-learn 1.8, skrub 0.8, numexpr 2.14.
All pinned through `devbox.json`. Timings are medians of five runs after a
warm-up. Every random input uses NumPy PCG64 seeded with `0`, so the *inputs*
and all structural results are deterministic; wall-clock and RSS naturally vary
between runs and machines (see Reproducibility notes).

---

## The harnesses

| Script | Question | Log |
|---|---|---|
| `structural_bench.py` | Does each rewrite fire, and how many ops does it remove? | `logs/structural_bench.log` |
| `order_bench.py` | Does the order of application change the result? | `logs/order_bench.log` |
| `walltime_bench.py` | Do the rewrites make pipelines faster end to end? | `logs/walltime_bench.log` |
| `accuracy_bench.py` | Do the stable fusions improve numerical accuracy? | `logs/accuracy_bench.log` |
| `memory_bench.py` | Do they reduce peak memory? | `logs/memory_bench.log` |
| `fusion_micro_bench.py` | What would a single-pass fused kernel buy? | `logs/fusion_micro_bench.log` |
| `fusion_region_bench.py` | Does CSE shrink the regions such a kernel could fuse? | `logs/fusion_region_bench.log` |
| `batch_pipeline_bench.py` | What do the rewrites contribute on a *batch* of pipelines? | `logs/batch_pipeline_bench.log` |

---

### `structural_bench.py` → report Table I

For each rewrite, builds a pipeline that triggers it and counts DAG operations
with that rewrite off, then on (everything else disabled). No dataset needed;
fully deterministic.

**Result:** all 18 fire; 46 → 25 operations, i.e. **−21 in isolation**.
(`zero_div` and `constant_folding` are opt-in, so the harness enables each
flag explicitly rather than relying on defaults.)

### `order_bench.py` → report §IV-B

Applies rewrite pairs in both orders on pipelines where one rewrite *enables*
the other (`abs(abs(x)·1)`, `neg(neg(x)+0)`).

**Result:** the outcome is order-dependent — `{2,3}` operations for the first,
`{1,3}` for the second. A fixpoint schedule removes the dependence at the cost
of extra traversals.

### `walltime_bench.py` → report §VI-C

Median wall-clock with all rewrites off vs on, over a synthetic 1M-row column
and two real columns (`california-housing` 20 640 rows, `covtype` 581 012 rows,
fetched via scikit-learn/OpenML — no credentials needed).

**Result (this run, `logs/walltime_bench.log`):** redundancy elimination
**1.35×** on the synthetic 1M column and **1.50×** on covtype; the `log1p` and
softmax fusions land between 0.86× and 1.04×, i.e. speed-neutral to slightly
slower. Timings vary by a few percent between runs — the fusions' value is
accuracy (below), not speed.

### `accuracy_bench.py` → report §VI-D

Compares naive vs fused evaluation against a high-precision reference on inputs
chosen to expose cancellation and overflow.

**Result:** naive `log(1+x)`/`exp(x)-1` at `x=1e-12` lose ~4 digits
(rel-err 8.9e-5); naive softmax and logsumexp on `[1000,1001,1002]` return
`NaN`/`inf`, while the fused forms are exact.

### `memory_bench.py` → report §VI-D

Peak RSS measured in a subprocess (`resource.getrusage`) at 20M rows, plus a
count of materialised intermediates.

**Result:** eliminations save **305 MB**; `log1p` fusion saves **153 MB** by
avoiding the `x+1` temporary; softmax fusion saves ≈0, because SciPy allocates
internally — DAG-level fusion into a library call cuts operator count but not
memory traffic.

### `fusion_micro_bench.py` → report Table II

Compares a per-operator NumPy chain against the same expression evaluated in a
single blocked pass (numexpr), over four expressions of 2–8 operators and input
sizes from 10³ to 10⁷. Verifies the two agree before timing.

**Result:** up to **7.83×** at 10⁷ rows with peak memory falling from 2–3× the
input array to exactly 1×. Below a crossover, fusion is a **regression** (as low
as 0.04×), and the crossover depends on chain length — roughly 5·10⁴ elements
for a four-operator chain but 10⁶ for a two-operator one. A single global
threshold would therefore be wrong.

### `fusion_region_bench.py` → report Table IV

Groups the optimised DAG into maximal fusable regions (elementwise operators
whose values are internal to the region) as the batch size grows, with CSE off
and on.

**Result:** without CSE mean region size grows with N (4.50 → 6.44); with CSE it
*shrinks* (4.00 → **2.33**), because deduplicated nodes acquire multiple
consumers and can no longer be absorbed. Distinct region *shapes* stay at 3
while the region count reaches 9, so a compiled-kernel cache would still hit
about two thirds of the time.

### `batch_pipeline_bench.py` → report Table III

A batch of eight pipeline variants over a shared preprocessing prefix,
mirroring the reference workload. Factors: CSE {off,on} × rewrites {off,on}.
Reports operation count, optimizer wall-clock, and peak RSS per cell.

Runs **two constructions** of the same batch:

- *varied* (default) — each variant carries a different redundant pattern
- *uniform* — every variant carries the same one

**Result:** both converge to 21 operations, but attribute it differently. Varied:
CSE removes 35, the rewrites a further **12** on top. Uniform: CSE removes 74,
the rewrites only **7** — because identical redundancy is deduplicated by CSE
before the rewrites run. The varied construction is the more realistic model of
agent output; the two are not a like-for-like comparison, since uniform
redundancy is simply *more* redundancy.

---

## Reproducibility notes

- **Deterministic:** `structural_bench`, `order_bench`, `fusion_region_bench`,
  and the operation counts in `batch_pipeline_bench` — these count DAG
  structure and should reproduce exactly.
- **Machine-dependent:** all timings, the fusion crossover points, and peak-RSS
  figures. Expect the same *shape* (crossover between 10⁴ and 10⁶; speedup
  growing with input size) rather than identical numbers.
- `walltime_bench.py` downloads `california-housing` and `covtype` on first run
  and caches them under scikit-learn's data home.
- Five failures in `tests/adapters/` are pre-existing and unrelated to the
  optimizer: they assert Rust kernels are selected and need `maturin develop`.
  The 667 optimizer tests pass.
