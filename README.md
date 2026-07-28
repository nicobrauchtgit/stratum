# Stratum Optimizer: Rule-Based Rewrites

**MLMMI SS2026 — Project 03**
Aiman Al-Hazmi, Adam Zalwowski, Mateusz Tomaszewski, Nicolas Kohl
Technische Universität Berlin

This fork adds **18 rule-based rewrites** to Stratum's logical optimizer, plus a
reproducible benchmark suite evaluating them. Upstream Stratum's own README is
preserved as [`README_stratum_upstream.md`](README_stratum_upstream.md).

- **Who wrote what:** [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md)
- **How to reproduce the numbers:** [`EXPERIMENTS.md`](EXPERIMENTS.md)
- **Benchmark discussion:** [`benchmarks/rewrites/REPORT.md`](benchmarks/rewrites/REPORT.md)

---

## Quick start

```bash
devbox run install     # nix env + venv + rust extension
devbox run test        # full test suite (667 optimizer tests)
devbox run test-rewrites   # just the rewrite tests
```

Then any benchmark, e.g.:

```bash
PYTHONPATH=$PWD .venv/bin/python benchmarks/rewrites/structural_bench.py
```

> **macOS note.** `lightgbm`/`xgboost` come from **nixpkgs**, not PyPI
> (`devbox.json`). Their macOS-ARM wheels link `@rpath/libomp.dylib` and search
> only Homebrew/MacPorts prefixes, so they fail to load in a nix-only
> environment. `devbox run install` writes a `.pth` file placing the nix builds
> on the venv path; they are deliberately **absent** from `pyproject.toml`'s
> `test` extra so `uv sync` cannot shadow them.

---

## Where our code lives

Everything we wrote is in three places. Nothing else in `stratum/` is ours.

### 1. The rewrites — `stratum/optimizer/_numeric_rewrites.py`

Each rewrite is a `(match, action)` pair wired together by `rewrite_pass`.
A matcher inspects a node and returns the matched ops or `None`; an action
rewires the DAG and returns the (possibly new) root.

The matcher/action helpers themselves (`match_two_op_chain`,
`match_identity_operation`, `replace_two_op_chain`, ...) are **pre-existing**
— see [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md#shared-infrastructure-we-did-not-write)
for who wrote what. The table below lists which helper each of our rewrites is
built from.

| Rewrite | Matcher | Action |
|---|---|---|
| `x*1`, `x+0`, `x-0`, `x/1`, `x**1` | `match_identity_operation` | `eliminate_single_op_chain_root_safe` |
| `x*0 → 0` | `match_identity_operation` | `fold_to_zero` |
| `x**0 → 1` | `match_identity_operation` | `fold_to_one` |
| `x-x → 0` | `match_self_subtract` | `fold_to_zero` |
| `0/x → 0` *(opt-in)* | `match_zero_div` | `fold_to_zero` |
| `abs(abs(x))` | `match_two_op_chain` | `make_replace_two_op_chain_root_safe` |
| `-(-x) → x` | `match_two_op_chain(..., innermost_first=True)` | `eliminate_two_op_chain_root_safe` |
| `exp(x)-1 → expm1` | `match_exp_minus_one` | replace-with-`EXPM1` |
| `log(1+x) → log1p` | `match_add_one_then_log` | replace-with-`LOG1P` |
| `log(sum(exp(x))) → logsumexp` | `match_log_sum_exp` | three-op replace |
| `exp(x)/sum(exp(x)) → softmax` | `match_softmax` | `replace_with_softmax` |
| constant folding | `match_constant_foldable` | `eliminate_constant_folding` |

Dispatch and feature flags: **`stratum/optimizer/_algebraic_rewrites.py`**
(`AlgebraicRewritesConfig` — one boolean per rewrite; `constant_folding` and `zero_div` are
opt-in (default `False`) because it pre-empts pattern-rewrite unit tests that
use constant fixtures).

Projection rewrites for dataframe DAGs (`select∘select`, `drop∘drop`) are in
**`stratum/optimizer/_projection_rewrites.py`**.

### 2. IR changes — `stratum/optimizer/ir/_numeric_ops.py`

Three operations were promoted from the `GENERIC` fallthrough into
`NumericOpType`, so matchers can key on a type instead of a wrapped function
object: **`POW`**, **`NEGATIVE`** and **`SUM`**.

`SUM` is the first *reduction* in an otherwise elementwise enum, so its
`process` branch **forwards `args`/`kwargs`** — the elementwise branches drop
them, which would silently turn `np.sum(x, axis=0)` into a whole-array sum.

### 3. Benchmarks — `benchmarks/rewrites/`

Eight harnesses; see [`EXPERIMENTS.md`](EXPERIMENTS.md) for what each measures
and how to run it, and `benchmarks/rewrites/logs/` for captured output.

---

## Headline findings

| Question | Answer |
|---|---|
| Do the rewrites fire? | All 18, −21 operations in isolation (Table I) |
| Do they speed things up? | Eliminations up to **1.35×**; fusions ≈1.0× |
| Then why fuse? | **Stability**: naive softmax/logsumexp overflow to `NaN`; naive `log1p`/`expm1` lose ~4 digits |
| Does order matter? | Yes — one rewrite enables another; a bad order leaves 3 ops where 2 suffice |
| What would give real speedup? | Single-pass fused kernels: **7.8×** and 2–3× less peak memory at 10M rows |
| Does it help on batches? | Partly — CSE does the bulk; our rewrites remove a further 12 of 33 ops on top of it |

A negative result worth stating: **CSE shrinks the regions a fused kernel could
absorb** (mean 6.44 → 2.33 operators at 8 pipelines), because deduplicated nodes
gain multiple consumers and can no longer be fused away. A batch-oriented kernel
should therefore target *shared intermediates*, not long chains.

---

## Testing

```bash
devbox run test            # 667 passed
devbox run test-rewrites   # rewrite tests only
```

Every rewrite ships tests for: it fires; it does *not* fire on near-misses
(`1/x`, `x**2`, `1**x`, mixed `abs`/`neg` chains, ndarray constants); a `False`
flag disables it; behaviour on chains, fan-out and NaN; and end-to-end numeric
equivalence against the unoptimised pipeline.

Five failures in `tests/adapters/` are pre-existing and unrelated — they assert
Rust kernels are selected and require `maturin develop` to have been run.
