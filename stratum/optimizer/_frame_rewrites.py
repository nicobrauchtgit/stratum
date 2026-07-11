from dataclasses import dataclass
from stratum.optimizer.ir._ops import Op
from stratum.optimizer.ir._projection_ops import ColumnSelectorOp, DropOp
from stratum.optimizer._op_utils import rewrite_pass
from stratum.optimizer._numeric_rewrites import eliminate_single_op_chain_root_safe
from stratum.utils._utils import start_time, log_time


@dataclass(frozen=True, slots=True)
class FrameRewritesConfig:
    select_select: bool = True
    drop_drop: bool = True


def _is_static_column_list(value) -> bool:
    """True for an explicit list of column names (no deferred selectors,
    no graph-fed OperandRefs)."""
    return isinstance(value, list) and all(isinstance(c, str) for c in value)


def match_consecutive_column_selects(op: Op):
    """Match ``select(s1) -> select(s2)`` for explicit column lists with
    ``set(s2) <= set(s1)``: then ``frame[s1][s2] == frame[s2]`` and the upstream
    select can be eliminated. Non-subset chains raise at runtime and must keep
    raising; deferred selectors can't be checked statically -- neither matches."""
    if not (isinstance(op, ColumnSelectorOp)
            and _is_static_column_list(op.selector)
            and len(op.outputs) == 1):
        return None
    op2 = op.outputs[0]
    if (isinstance(op2, ColumnSelectorOp)
            and _is_static_column_list(op2.selector)
            and set(op2.selector) <= set(op.selector)):
        return (op,)
    return None


def _is_pure_columns_drop(op: Op) -> bool:
    """True for the exact shape ``drop(columns=[...])`` -- no positional args,
    no other kwargs, static string column list."""
    return (isinstance(op, DropOp)
            and not op.args
            and set(op.kwargs) == {"columns"}
            and _is_static_column_list(op.kwargs["columns"]))


def match_consecutive_drops(op: Op):
    """Match ``drop(columns=c1) -> drop(columns=c2)`` with disjoint column sets:
    overlapping drops raise KeyError sequentially, so a fused union would
    silently swallow that error."""
    if not (_is_pure_columns_drop(op) and len(op.outputs) == 1):
        return None
    op2 = op.outputs[0]
    if (_is_pure_columns_drop(op2)
            and not set(op.kwargs["columns"]) & set(op2.kwargs["columns"])):
        return (op, op2)
    return None


def absorb_drop_into_downstream(op1: Op, op2: Op, root: Op) -> Op:
    """Fold c1 into the downstream drop, then eliminate the upstream one --
    mutating the survivor lets us reuse ``eliminate_single_op_chain_root_safe``."""
    op2.kwargs = {"columns": list(op1.kwargs["columns"]) + list(op2.kwargs["columns"])}
    return eliminate_single_op_chain_root_safe(op1, root)


eliminate_consecutive_selects = rewrite_pass(
    match_consecutive_column_selects,
    eliminate_single_op_chain_root_safe,
)

fuse_consecutive_drops = rewrite_pass(
    match_consecutive_drops,
    absorb_drop_into_downstream,
)


def frame_rewrites(root: Op, config: FrameRewritesConfig) -> Op:
    """Run all enabled frame-operator rewrites, one pass per rewrite."""
    start = start_time()
    if config.select_select:
        root = eliminate_consecutive_selects(root)
    if config.drop_drop:
        root = fuse_consecutive_drops(root)
    log_time("frame_rewrite", start)
    return root
