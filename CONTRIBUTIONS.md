# Contributions

MLMMI SS2026 — Project 03. GitHub handles in brackets.

Authorship below is traceable in `git log` and in the upstream pull requests
listed per rewrite. Six rewrites were merged into `deem-data/stratum` directly
and reached this fork through upstream `main`; the remainder were integrated
here from the team's individual branches.

---

## Summary

| Member | Rewrites | Other |
|---|---|---|
| Aiman Al-Hazmi (`aimanalhazmi`) | `abs(abs(x))`, `x+0`, `exp(x)-1 → expm1`, `0/x → 0` | report |
| Mateusz Tomaszewski (`matit02`) | `x-0`, `x**0 → 1` (+ `POW` enum), `log(sum(exp(x))) → logsumexp`, constant folding | report |
| Adam Zalwowski (`Sandi077`) | `x*0 → 0`, `log(1+x) → log1p`, `select∘select`, `drop∘drop` | report |
| Nicolas Kohl (`nicobrauchtgit`) | `x/1`, `x**1`, `-(-x)`, `exp/sum(exp) → softmax` (+ `NEGATIVE`/`SUM` enum) | integration, benchmark suite, report |

---

## Per-member detail

### Aiman Al-Hazmi (`aimanalhazmi`)

- **`abs(abs(x)) → abs(x)`** — upstream PR #105.
  `stratum/optimizer/_numeric_rewrites.py`: `eliminate_abs_abs`, built on
  `match_two_op_chain` + `_replace_with_abs`.
- **`x+0 → x`, `0+x → x`** — upstream PR #120 (merged as #123).
  `eliminate_add_zero`.
- **`exp(x)-1 → expm1(x)`** — upstream PR #121.
  `match_exp_minus_one` + `_replace_with_expm1`.
- **`0/x → 0`** — PR #154. Opt-in, since `0/0` is `NaN` and folding skips
  evaluation of the divisor.
- Tests: `stratum/tests/logical_optimizer/algebraic_rewrites/test_numeric.py`
  (abs/add-zero/expm1 cases).

### Mateusz Tomaszewski (`matit02`)

- **`x-0 → x`** — upstream PR #138 (merged as #134). `eliminate_identity_subtract`.
- **`x**0 → 1`** — PR #139. `eliminate_pow_zero` + `fold_to_one`.
  **Also added `POW` to `NumericOpType`** (`stratum/optimizer/ir/_numeric_ops.py`),
  the representation change discussed in the report's §IV-A.
- **`log(sum(exp(x))) → logsumexp(x)`** — PR #142.
  `match_log_sum_exp`, `_logsumexp`, `make_replace_three_op_chain_root_safe`.
- **Constant folding** — PR #143. `match_constant_foldable`,
  `eliminate_constant_folding`. Default `False`; see `AlgebraicRewritesConfig`.
- Commits: `Add log-sum-exp algebraic rewrite for numeric ops`,
  `Add constant folding compile-time evaluation for numeric ops`,
  `[Rewrite] Power-of-zero annihilator (x**0 -> 1) + POW NumericOpType`.

### Adam Zalwowski (`Sandi077`)

- **`x*0 → 0`, `0*x → 0`** — upstream PR #130. `eliminate_any_mul_zero` +
  `fold_to_zero`.
- **`log(1+x) → log1p(x)`** — PR #131. `match_add_one_then_log`,
  `rewrite_log_plus_one`.
- **Consecutive `select` fusion** — PR #132.
  `stratum/optimizer/_projection_rewrites.py`: `match_consecutive_select`,
  `eliminate_redundant_select_action`.
- **Consecutive `drop` fusion** — PR #148. Same file:
  `match_consecutive_drop`, `fuse_consecutive_drop_action`,
  `_extract_drop_columns`.
- Commits: `[Rewrite] fuse consecutive column selects`,
  `[Rewrite] fuse consecutive column drops`,
  `[Rewrite] fuse log(x+1) -> log1p(x)`.

### Nicolas Kohl (`nicobrauchtgit`)

**Rewrites** — all in `stratum/optimizer/_numeric_rewrites.py`:

- **`x/1 → x`** — PR #144 (merged upstream as #149). `eliminate_div_by_one`.
- **`x**1 → x`** — PR #145. `eliminate_pow_by_one`, expressed via
  `match_identity_operation(NumericOp, NumericOpType.POW, 1, reversed=False)`.
  Originally matched a raw `BinOp`; ported onto the `POW` enum after #139
  landed (conflict *G1*, report §IV-A).
- **`-(-x) → x`** — PR #146. `eliminate_neg_neg`. Promoted `NEGATIVE` into
  `NumericOpType` and added the `innermost_first` flag to `match_two_op_chain`,
  which fixes odd-length chains (`neg^3`, `neg^5`) that otherwise abort the
  traversal.
- **`exp(x)/sum(exp(x)) → softmax(x)`** — PR #147. `match_softmax`,
  `replace_with_softmax`. Promoted `SUM` into `NumericOpType`, with a
  `process` branch that forwards `args`/`kwargs` (see README §2).

**IR / shared infrastructure** — `stratum/optimizer/ir/_numeric_ops.py`:

- `NEGATIVE` and `SUM` enum members, extraction-map entries and `process` branches.
- Scalar guard widened from `(int, float)` to `numbers.Real` in
  `match_identity_operation`, so numpy scalar constants (`np.int64(1)`) are not
  silently skipped. Affects every identity rewrite, not only `x**1`.
- `isinstance` guard ordered before the `== 2` comparison in
  `extract_numeric_op`, fixing a crash on `df ** np.array([...])`.

**Integration** — reconciling four contributors' branches onto one base,
including the `POW`/`x**1` and `SUM`/`logsumexp` representation conflicts
described in report §IV-A and §IV-C.

**Benchmarks** — `benchmarks/rewrites/` (all eight harnesses; see
`EXPERIMENTS.md`) and the accompanying analysis in `REPORT.md`.

**Report** — §III entries for `x**1`, `-(-x)` and softmax; §IV Technical
Challenges; §V correctness arguments for the same three; §VI Performance
Evaluation.

---

## Shared infrastructure we did *not* write

Most of the matcher/action helpers our rewrites are built from predate this
project. They come from the Stratum maintainer and from earlier contributors,
and are listed here so the boundary of our own work is unambiguous. Each row is
traceable with `git log -S"def <name>" -- stratum/optimizer/_numeric_rewrites.py`.

| Helper | Introduced by | Commit |
|---|---|---|
| `rewrite_pass`, `topological_iterator`, `Op` IR, CSE, linearisation, scheduler | Elias Strauss (maintainer) | various |
| `match_two_op_chain`, `eliminate_two_op_chain`, `eliminate_two_op_chain_root_safe` | Elias Strauss | `3a4e2be` *"rewrite API for all future rewrites"* |
| `replace_two_op_chain`, `make_replace_two_op_chain_root_safe` | Tim Schuster | `e9e60cc` (`sqrt(x**2) → abs(x)`, #59) |
| `match_identity_operation`, `eliminate_single_op_chain(_root_safe)` | Aram164 | `a7b7dc6` (`x*1 → x`, #92) |
| `_is_scalar_const` (ndarray guard) | Elias Strauss | `1ab4687` |
| `_matches_scalar_const`, `match_two_op_chain(match1=, match2=)` | Elias Strauss | `0d36fa4` |

Neither Tim Schuster nor Aram164 is a member of this project team; their
rewrites were merged upstream before ours and we build on their helpers.

**What the team added to that shared layer:**

| Change | Member | Why |
|---|---|---|
| `fold_to_zero` | Adam (`a51ce41`, #130) | `x*0` yields a constant, not an input — needed a new action shape |
| `fold_to_one` | Mateusz (`06816dc`, #135) | same, for `x**0` |
| `POW` in `NumericOpType` | Mateusz (`06816dc`) | representation change; see report §IV-A |
| `NEGATIVE`, `SUM` in `NumericOpType` | Nicolas | lets matchers key on a type instead of `GENERIC` + a function object |
| `innermost_first` on `match_two_op_chain` | Nicolas | odd-length involution chains; opt-in so idempotent rewrites are unaffected |
| `_is_scalar_const` widened to `numbers.Real` | Nicolas | numpy scalar constants were silently skipped by *every* identity rewrite |

The last two are the cases where a change to a shared helper affected rewrites
other than the one motivating it — the merge-conflict surface described in
report §IV-C.

## Environment

`devbox.json` / `pyproject.toml` (nix-provided `lightgbm`/`xgboost`) — Nicolas.
