# Rule-Based Rewrites for Stratum's Logical Optimizer

**MLMMI SS2026 — Project #3 (Stratum Optimizer: Rule-Based Rewrites)**
Authors: Nicolas (nicobrauchtgit), Mateusz (matit02), Sandi (Sandi077), Aiman (aimanalhazmi)
Base: `deem-data/stratum@a51ce41`. Integration base: `nicobrauchtgit/stratum:main`.

---

## Abstract

We extend Stratum's logical optimizer [Stratum, arXiv:2603.03589] with 16 rule-based
rewrites over its operator-IR DAG: algebraic identity/annihilator elimination, double-inverse
elimination, numerically-stable fusions (`log1p`, `expm1`, `softmax`, `logsumexp`), constant
folding, and consecutive projection (`select`/`drop`) fusion. Each rewrite is a
`(matcher, action)` pair wired into a configurable pass pipeline. We describe the
implementation, the technical challenges — most notably the **phase-ordering problem**
(the order in which rewrites are applied changes the optimized DAG) and an IR
representation change (`POW`) that forced us to reconcile two rewrites — how we verified
correctness (per-rewrite unit tests, a structural op-count harness, and numeric
equivalence), and a benchmark study over synthetic and real tabular data. We find that
redundancy-eliminating rewrites yield modest-but-real end-to-end speedups (up to ~1.35×
on redundant pipelines at scale), that the fusion rewrites are essentially speed-neutral
but improve numerical stability, and that rewrite order **deterministically changes the
result** when one rewrite enables another (e.g. `abs(abs(x)·1)`: 2 vs 3 ops; `neg(neg(x)+0)`:
1 vs 3 ops), which a fixpoint schedule resolves at extra cost.

---

## 1. Introduction

Stratum fuses batches of (often agent-generated, highly redundant) skrub pipelines into a
single operator DAG and applies logical optimizations before lowering to physical
operators [Stratum §4.2]. Figure 2 of the Stratum paper shows that agent search produces
pipelines where the median iteration changes only 16% of code lines — i.e. massive
cross-pipeline redundancy — which is exactly the setting where semantics-preserving
rewrites pay off.

Our project contributes a family of **numeric algebraic rewrites** and **projection-fusion
rewrites** to Stratum's rule-based optimizer. Each rewrite consists of (i) a *pattern
matcher* over the typed operator IR, (ii) a *rewrite action* that rewires the DAG, and
(iii) unit tests. This report documents the rewrites, the engineering challenges,
correctness verification, and a performance study focused on two questions the Stratum
authors flagged as important: *do the rewrites help*, and *does the order of applying them
matter*.

## 2. Background: the operator IR

Stratum converts a skrub DataOp DAG into a typed operator IR (`convert_to_ops`), extracts
specialized frame/numeric operators, runs Op-level CSE, unrolls choices, and then applies
`algebraic_rewrites(root, config)` — a sequence of single-pass rewrites — before
linearization and scheduling. Numeric expressions become `NumericOp`s carrying a
`NumericOpType` (`ADD`, `MULTIPLY`, `DIVIDE`, `LOG`, `EXP`, `ABS`, `SQUARE`, …), a scalar
`constant`, an optional `opt_operand` (for var⊗var binaries), and a `reversed` flag;
unknown numpy ufuncs fall through to `GENERIC` carrying the `func`. Column selection lowers
to `GetItemOp` (raw `df[cols]`) or `ColumnSelectorOp` (`skb.select`, once the frame type is
resolved); `df.drop(...)` lowers to `DropOp`. Each rewrite is built with the helper
`rewrite_pass(match, action)`, and reuses shared rewiring helpers
(`eliminate_single_op_chain_root_safe`, `make_replace_two_op_chain_root_safe`, …).

## 3. Implemented rewrites

All 16 rewrites are exposed as boolean flags on `AlgebraicRewritesConfig`
(default `True`, except `constant_folding`, see §4.2) and dispatched by
`algebraic_rewrites()`. Six were already upstream when we started (merged under other PRs);
we integrated the remaining ten. Grouped by kind:

**A. Identity elimination** (drop a no-op node; `reversed`-aware where non-commutative)
| Rewrite | Matcher | Notes |
|---|---|---|
| `x*1 → x`, `1*x → x` | `match_identity_operation(MULTIPLY, 1)` | direction-agnostic |
| `x+0 → x`, `0+x → x` | `match_identity_operation(ADD, 0)` | |
| `x-0 → x` | `match_identity_operation(SUBTRACT, 0, reversed=False)` | only `x-0` |
| `x/1 → x` | `match_identity_operation(DIVIDE, 1, reversed=False)` | not `1/x` |
| `x**1 → x` | `match_identity_operation(POW, 1, reversed=False)` | see §4.1 |

**B. Annihilators**
| `x*0 → 0`, `0*x → 0` | `match_identity_operation(MULTIPLY, 0)` + `fold_to_zero` → `ValueOp(0)` |
| `x**0 → 1` | `match_identity_operation(POW, 0)` + `fold_to_one` → `ValueOp(1)` |

**C. Double-inverse elimination** (two-op chain → collapse/replace)
| `neg(neg(x)) → x` | `match_two_op_chain_by_func(np.negative)` (innermost-first) |
| `abs(abs(x)) → abs(x)` | `match_two_op_chain(ABS, ABS)` |
| `log(exp(x)) → x`, `exp(log(x)) → x`, `sqrt(x²) → |x|`, `log1p(expm1)`, `expm1(log1p)` | pre-existing base |

**D. Numerically-stable fusions** (sub-DAG → single op)
| `exp(x)-1 → expm1(x)` | `match_exp_minus_one` → `EXPM1` |
| `log(x+1) → log1p(x)` | `match_add_one_then_log` (scalar `+1`, single consumer) → `LOG1P` |
| `log(sum(exp(x))) → logsumexp(x)` | `EXP→GENERIC(np.sum)→LOG` chain → stable `logsumexp` |
| `exp(x)/sum(exp(x)) → softmax(x)` | diamond match (shared `EXP`, strict fan-out) → `scipy.special.softmax` |

**E. Constant folding** — `f(constants…) → ValueOp(result)`; matches a `NumericOp` whose
variable inputs are all `ValueOp` (unary, var-const, or var-var). *Opt-in* (§4.2).

**F. Projection fusion** (dataframe DAGs)
| `df[c1][c2] → df[c2]` (`c2 ⊆ c1`) | `match_consecutive_select` over `GetItemOp` |
| `drop(c1);drop(c2) → drop(c1∪c2)` (disjoint) | `match_consecutive_drop` over `DropOp` (order-preserving union) |

## 4. Technical challenges

### 4.1 Phase ordering and an IR representation change (`POW`)

The sharpest challenge was the interaction between `x**0 → 1` and `x**1 → x`. The original
`x**1` rewrite matched a **raw `BinOp`** with `operator.pow`, *because `POW` was not a
`NumericOpType`*. The `x**0` rewrite, however, **added `POW` to the enum** so that `x**n`
now lowers to `NumericOp(type=POW)`. Merged naively, `x**1`'s matcher would silently stop
firing (it looked for a `BinOp` that no longer exists). We resolved this by (a) ordering
the merge so the `POW`-enum change lands first, and (b) rewriting `x**1` as a first-class
identity, `match_identity_operation(POW, 1, reversed=False)` — the *same* helper as `x/1`.
This is a representation-level phase-ordering constraint: a rewrite that changes how nodes
are represented must precede rewrites that pattern-match on the old representation. §6.2
quantifies the more general, DAG-level phase-ordering effect.

### 4.2 Constant folding interacts with every pattern rewrite

Constant folding is correct — it only folds genuine compile-time constants (`ValueOp`), and
on real pipelines with `skb.var` inputs it leaves pattern rewrites (softmax, log1p, …)
untouched (verified). But its default-`True` setting **preempts** every pattern-rewrite
*unit test* that uses a constant fixture (e.g. `as_data_op(3)`): the fixture folds to a
constant before the pattern can match. Rather than couple ~74 tests (and all future ones)
to `constant_folding=False`, we made it **opt-in (default `False`)** in the shared base;
benchmarks enable it explicitly. This keeps per-rewrite measurement and testing isolated
while leaving the rewrite itself unchanged.

### 4.3 Fusion rewrites need frame-type resolution

`select`/`drop` fusion only fire once the input's frame *type* is known. On a bare untyped
pipeline, `df.drop(columns=…)` stays a generic `MethodCallOp` and is never lowered to
`DropOp`, so `match_consecutive_drop` cannot match. Passing `env` (as `evaluate()` does)
resolves the schema, `.drop` lowers to `DropOp`, and fusion fires (3→2 ops). We surface
this in the benchmark so the rewrite is measured in a realistic, typed setting.

### 4.4 Integration mechanics

Every numeric rewrite edits the same three anchors of `_algebraic_rewrites.py` (import
block, config dataclass, dispatch body), so combining independent PRs produced **mechanical
merge conflicts** that are resolved as an additive union. Some contributed branches were
also stale (based ~30 commits behind) or carried stray artifacts; we rebased them onto the
current base and reconstructed the semantic change where a clean cherry-pick was impossible.

### 4.5 A missed optimization we found

`y * exp(0)` remains `y * 1` (not `y`): constant folding turns `exp(0)` into `ValueOp(1)`,
but `eliminate_identity_operation` only matches a **scalar literal** `*1`, not a folded
`ValueOp(1)` *operand*. Extending the identity matcher to accept `ValueOp`-1 operands would
let folding and identity compose here — a concrete follow-up.

## 5. Correctness verification

We verified correctness at three levels:

1. **Per-rewrite unit tests.** Each rewrite ships tests covering: it fires on the target
   pattern; it does *not* fire on near-misses (`1/x`, `x/2`, `x**2 → SQUARE`, single `neg`,
   different funcs in a chain); a `flag=False` disables it; behavior on chains, fan-out, and
   NaN inputs; and end-to-end numeric results. The full logical-optimizer suite is **543
   tests, all passing** on the integrated base.
2. **Structural check** (`structural_bench.py`): for each rewrite, build a triggering
   pipeline and confirm the DAG op-count strictly decreases (off → on). All 16 fire; −19
   ops in isolation.
3. **Semantic equivalence.** For pattern rewrites we assert the optimized pipeline produces
   the same numeric output as the unoptimized one; the stability fusions (`log1p`, `expm1`,
   `logsumexp`, `softmax`) are validated against their `numpy`/`scipy` reference
   implementations (which are *more* accurate than the naive form for small/large inputs —
   an accuracy improvement, not just a rewrite).

Near-miss/negative tests are essential: e.g. `x/1` must use `reversed=False` so `1/x`
(non-commutative) is untouched, and `x**2` must remain `SQUARE`.

## 6. Performance benchmarks

### 6.1 Experimental setup

Hardware: Apple M3 Pro (12 cores), 36 GB RAM, macOS (arm64). Software: Python 3.12,
NumPy 2.3, pandas 3.0, SciPy 1.16, scikit-learn 1.8, skrub 0.8. We report the **median of 5
timed runs** after one warm-up, timing full pipeline execution (`optimize()` +
`SequentialScheduler.compute`). Rewrites are toggled via `OptConfig(algebraic_rewrites=…)`
and `AlgebraicRewritesConfig`. Data of **different kinds and sizes**: a reproducible
synthetic 1M-row column, and two real tabular columns from OpenML/sklearn —
`california-housing MedInc` (20,640 rows) and `covtype Elevation` (581,012 rows). (Kaggle
was suggested but requires credentials on the runner; OpenML/sklearn provides equivalent
real tabular data without authentication.) Three reproducible harnesses live in
`benchmarks/rewrites/`: `structural_bench.py`, `order_bench.py`, `walltime_bench.py`.

### 6.2 Q1 — Do the rewrites improve the pipeline?

*Structural (deterministic).* All 16 rewrites fire and reduce the DAG (Table, §5). This is
the guaranteed win: fewer operators to schedule, materialize, and execute.

*Wall-clock (rewrites OFF vs ON, median of 5):*

| pipeline | synthetic 1M | california 20K | covtype 581K |
|---|---|---|---|
| redundant identities (`x/1,**1,*1,+0,-0,neg-neg`) | **1.23×** | 1.04× | **1.35×** |
| `log1p` fusion (`log(x+1)`) | 0.92× | 1.03× | 0.98× |
| `softmax` fusion (`exp/sum(exp)`) | 0.93× | 0.98× | 1.01× |

**Findings.** (1) *Redundancy elimination is where the speed win is*: stacking six
redundant algebraic ops on a column and removing them all gives ~1.2–1.35× end-to-end at
scale — and this compounds with pipeline redundancy, which is precisely the agentic
workload Stratum targets. (2) *Fusions are speed-neutral* (~1.0×): `log1p`/`softmax` fire
(op-count drops) but the fused kernel does equivalent work; their value is **numerical
stability/accuracy**, which a runtime benchmark does not capture. This mirrors the Stratum
paper's ablation, where logical rewrites alone contribute a bounded speedup (≈2.2× via
CSE + rewrites) and the larger gains come from operator selection and parallelism.

### 6.3 Q2 — Does the order of applying (merging) rewrites matter?

Each pass runs **once** in the dispatch sequence of `algebraic_rewrites()` — and that
sequence is effectively the order the rewrites were merged. When rewrite *A* removes a node
that blocks rewrite *B*'s pattern, *B* only fires if it runs **after** *A*. We enumerate all
orderings of the relevant passes and record the final op-count (`order_bench.py`):

| pipeline | orders | final op-counts | fixpoint | order-sensitive |
|---|---|---|---|---|
| `abs(abs(x)·1)` | identity_op, abs_abs | **{2, 3}** | 2 | **YES** |
| `neg(neg(x)+0)` | add_zero, neg_neg | **{1, 3}** | 1 | **YES** |
| `abs(abs(x·1)·1)` | identity_op, abs_abs | {2, 3} | 2 | **YES** |
| `(x/1)**1 → neg(neg)` | div, pow, neg (disjoint) | {1} | 1 | no |

The "good" order (`identity_op → abs_abs`) reduces `abs(abs(x)·1)` to 2 ops; the "bad" order
(`abs_abs → identity_op`) leaves 3, because `abs_abs` ran while a `·1` still separated the
two `abs` nodes and never got a second chance. Disjoint-pattern rewrites are confluent
(order-independent).

*Wall-clock of the phase-ordering effect* (`abs(abs(x)·1)`, median of 5):

| data | good (2 ops) | bad (3 ops) | penalty | fixpoint |
|---|---|---|---|---|
| synthetic 1M | 8.30 ms | 8.19 ms | ~1.0× | 7.88 ms |
| california 20K | 6.77 ms | 6.72 ms | ~1.0× | 6.80 ms |
| covtype 581K | 7.74 ms | 9.05 ms | **1.17×** | 9.00 ms |

**Findings.** Order changes the *result* deterministically (a missed rewrite = a surviving
operator); the *wall-clock* penalty scales with how expensive the missed operator is — a
single extra `abs` is within noise at 20K–1M but reaches 1.17× at 581K, and would compound
for heavier/more numerous missed rewrites. Running rewrites **to a fixpoint** (repeat the
set until the DAG stops changing) makes the outcome order-independent, at the cost of extra
passes. This is a concrete instance of the Stratum paper's statement that *"rewrite ordering
is workload-dependent"*, and it argues for either a principled ordering (enabling rewrites
before the rewrites they enable) or fixpoint iteration.

## 7. Discussion & conclusion

We integrated 16 semantics-preserving rewrites into Stratum's optimizer, each with a
matcher, an action, and tests, on a single benchmarked base. Structurally every rewrite
reduces the DAG; empirically the redundancy-eliminating rewrites give small but real,
scale-dependent speedups on the redundant pipelines Stratum targets, while the fusion
rewrites trade speed-neutrality for numerical stability. Our central finding on ordering:
the current single-pass optimizer is **order-sensitive for enabling interactions** — merge
order can leave optimizations on the table — and a fixpoint schedule (or an
enabling-aware ordering) removes the dependence. Future work: match folded `ValueOp`
constants in the identity rewrites (§4.5), extend `select`/`drop` fusion to
`ColumnSelectorOp`, add an accuracy benchmark for the stability fusions, and integrate the
rewrites with CSE/pushdown ordering as the paper suggests (delaying projection pushdown for
higher CSE opportunities).

## Appendix — reproducing

```bash
git switch -c bench-run nicobrauchtgit/stratum:bench   # base + benchmarks/rewrites/
PYTHONPATH=$PWD .venv/bin/python benchmarks/rewrites/structural_bench.py
PYTHONPATH=$PWD .venv/bin/python benchmarks/rewrites/order_bench.py
PYTHONPATH=$PWD .venv/bin/python benchmarks/rewrites/walltime_bench.py
```
Rewrites and tests live in `stratum/optimizer/_numeric_rewrites.py`,
`stratum/optimizer/_projection_rewrites.py`, `stratum/optimizer/_algebraic_rewrites.py`,
and `stratum/tests/logical_optimizer/`.
