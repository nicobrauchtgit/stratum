import unittest
import pandas as pd
import polars as pl
import skrub
import stratum as st
from stratum._config import FLAGS
from stratum.optimizer._optimize import optimize, OptConfig
from stratum.optimizer._frame_rewrites import FrameRewritesConfig
from stratum.optimizer.ir._projection_ops import ColumnSelectorOp, DropOp
from skrub import selectors as sel


def run_plan(dag):
    """Execute a linearized plan; fit_transform so deferred selectors resolve."""
    cache = {}
    for op in dag:
        ins = [cache[id(i)] for i in op.inputs]
        cache[id(op)] = op.process("fit_transform", ins)
    return cache[id(dag[-1])]


def make_frame():
    return pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5], "c": ["x", "y"], "d": [5, 6]})


NO_FRAME_REWRITES = dict(config=OptConfig(frame_rewrites=False))


class TestSelectSelectFusion(unittest.TestCase):

    def test_select_select_fuses_and_is_equivalent(self):
        x = st.as_data_op(make_frame())
        t = x.skb.select(["a", "b", "d"]).skb.select(["d", "a"])

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 2)                       # source + one select
        self.assertIsInstance(dag[1], ColumnSelectorOp)
        self.assertEqual(dag[1].selector, ["d", "a"])

        # Equivalence: optimized result == unoptimized result (incl. column order)
        dag_ref, *_ = optimize(t, **NO_FRAME_REWRITES)
        self.assertEqual(len(dag_ref), 3)
        pd.testing.assert_frame_equal(run_plan(dag), run_plan(dag_ref))

    def test_select_select_preserves_downstream_order(self):
        """select(['a','b','d']) then select(['d','a']) must yield ['d','a'] —
        the order-unsafe `s1 & s2` selector conjunction would yield ['a','d']."""
        x = st.as_data_op(make_frame())
        t = x.skb.select(["a", "b", "d"]).skb.select(["d", "a"])

        dag, *_ = optimize(t)
        result = run_plan(dag)
        self.assertEqual(list(result.columns), ["d", "a"])

    def test_triple_select_chain_fuses_to_one(self):
        x = st.as_data_op(make_frame())
        t = (x.skb.select(["a", "b", "c", "d"])
              .skb.select(["a", "b", "d"])
              .skb.select(["b"]))

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 2)
        self.assertEqual(dag[1].selector, ["b"])
        self.assertEqual(list(run_plan(dag).columns), ["b"])

    def test_select_not_fused_when_not_subset(self):
        """select(['a','b']) then select(['a','c']): 'c' was projected away by the
        first select, so the fused-against-original form would resurrect it and
        silently change results. The subset guard must reject the pair, leaving
        both selects in place.

        This pipeline is invalid (skrub's eager preview rejects it at authoring
        time), so we disable eager evaluation to drive the real
        construct -> extract -> optimize path and assert the guard holds there —
        rather than hand-building the IR."""
        skrub.set_config(eager_data_ops=False)
        try:
            x = st.as_data_op(make_frame())
            t = x.skb.select(["a", "b"]).skb.select(["a", "c"])
            dag, *_ = optimize(t)
        finally:
            skrub.set_config(eager_data_ops=True)

        selects = [op for op in dag if isinstance(op, ColumnSelectorOp)]
        self.assertEqual(len(selects), 2)                   # both survive, not fused
        # The behavior the guard preserves: the unfused plan raises on the missing
        # column. A wrongly-fused plan would silently succeed instead.
        with self.assertRaises(ValueError):
            run_plan(dag)

    def test_select_deferred_selector_not_fused(self):
        """numeric() cannot be subset-checked statically — leave the chain alone."""
        x = st.as_data_op(make_frame())
        t = x.skb.select(["a", "b", "c"]).skb.select(sel.numeric())

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 3)
        result = run_plan(dag)
        self.assertEqual(list(result.columns), ["a", "b"])

    def test_select_not_fused_with_second_consumer(self):
        """The intermediate select feeds a second consumer — eliminating it would
        change what that consumer sees."""
        x = st.as_data_op(make_frame())
        s1 = x.skb.select(["a", "b"])
        s2 = s1.skb.select(["a"])
        t = s2.assign(extra=s1["b"])

        dag, *_ = optimize(t)
        selects = [op for op in dag if isinstance(op, ColumnSelectorOp)]
        self.assertEqual(len(selects), 2)                   # both survive
        result = run_plan(dag)
        self.assertEqual(list(result.columns), ["a", "extra"])
        self.assertEqual(list(result["extra"]), [3.5, 4.5])

    def test_select_fusion_disabled(self):
        x = st.as_data_op(make_frame())
        t = x.skb.select(["a", "b"]).skb.select(["a"])
        config = OptConfig(frame_rewrite_config=FrameRewritesConfig(select_select=False))

        dag, *_ = optimize(t, config=config)
        self.assertEqual(len(dag), 3)

    def test_select_select_empty_frame(self):
        x = st.as_data_op(pd.DataFrame({"a": [], "b": []}))
        t = x.skb.select(["a", "b"]).skb.select(["b"])

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 2)
        result = run_plan(dag)
        self.assertEqual(list(result.columns), ["b"])
        self.assertEqual(len(result), 0)

    def test_select_select_polars(self):
        x = st.as_data_op(pl.DataFrame({"a": [1, 2], "b": [3.5, 4.5], "c": ["x", "y"]}))
        t = x.skb.select(["a", "b"]).skb.select(["b"])

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 2)
        FLAGS.force_polars = True
        try:
            result = run_plan(dag)
        finally:
            FLAGS.force_polars = False
        self.assertEqual(result.columns, ["b"])
        self.assertEqual(result["b"].to_list(), [3.5, 4.5])


class TestDropDropFusion(unittest.TestCase):

    def test_drop_drop_fuses_and_is_equivalent(self):
        x = st.as_data_op(make_frame())
        t = x.drop(columns=["c"]).drop(columns=["b"])

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 2)                       # source + one drop
        self.assertIsInstance(dag[1], DropOp)
        self.assertEqual(dag[1].kwargs["columns"], ["c", "b"])

        dag_ref, *_ = optimize(t, **NO_FRAME_REWRITES)
        self.assertEqual(len(dag_ref), 3)
        pd.testing.assert_frame_equal(run_plan(dag), run_plan(dag_ref))

    def test_triple_drop_chain_fuses_to_one(self):
        x = st.as_data_op(make_frame())
        t = x.drop(columns=["c"]).drop(columns=["b"]).drop(columns=["d"])

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 2)
        self.assertEqual(dag[1].kwargs["columns"], ["c", "b", "d"])
        self.assertEqual(list(run_plan(dag).columns), ["a"])

    def test_drop_overlap_not_fused(self):
        """drop(['b','c']) then drop(['c']): the second drop hits an
        already-dropped column and raises KeyError sequentially, while a fused
        drop(['b','c','c']) against the original frame would silently succeed.
        The disjointness guard must reject the pair, leaving both drops in place.

        The overlapping pipeline is invalid (skrub's eager preview rejects it), so
        we disable eager evaluation to exercise the real optimize path and assert
        the guard holds there."""
        skrub.set_config(eager_data_ops=False)
        try:
            x = st.as_data_op(make_frame())
            t = x.drop(columns=["b", "c"]).drop(columns=["c"])
            dag, *_ = optimize(t)
        finally:
            skrub.set_config(eager_data_ops=True)

        drops = [op for op in dag if isinstance(op, DropOp)]
        self.assertEqual(len(drops), 2)                     # both survive, not fused
        # The behavior the guard preserves: the unfused plan raises KeyError on
        # the already-dropped column. A wrongly-fused union would silently succeed.
        with self.assertRaises(KeyError):
            run_plan(dag)

    def test_drop_not_fused_with_second_consumer(self):
        """The intermediate drop feeds a second consumer — absorbing it into the
        downstream drop would change what that consumer sees, so it must not fuse."""
        x = st.as_data_op(make_frame())
        d1 = x.drop(columns=["c"])
        d2 = d1.drop(columns=["b"])
        t = d2.assign(from_d1=d1["b"])

        dag, *_ = optimize(t)
        drops = [op for op in dag if isinstance(op, DropOp)]
        self.assertEqual(len(drops), 2)                     # both survive
        result = run_plan(dag)
        self.assertEqual(list(result["from_d1"]), [3.5, 4.5])

    def test_drop_axis_variant_not_fused(self):
        """drop('c', axis=1) has positional args — shape not covered, leave alone."""
        x = st.as_data_op(make_frame())
        t = x.drop("c", axis=1).drop(columns=["b"])

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 3)
        result = run_plan(dag)
        self.assertEqual(list(result.columns), ["a", "d"])

    def test_drop_fusion_disabled(self):
        x = st.as_data_op(make_frame())
        t = x.drop(columns=["c"]).drop(columns=["b"])
        config = OptConfig(frame_rewrite_config=FrameRewritesConfig(drop_drop=False))

        dag, *_ = optimize(t, config=config)
        self.assertEqual(len(dag), 3)

    def test_mixed_select_drop_not_fused(self):
        x = st.as_data_op(make_frame())
        t = x.skb.select(["a", "b", "c"]).drop(columns=["c"])

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 3)
        result = run_plan(dag)
        self.assertEqual(list(result.columns), ["a", "b"])

    def test_drop_mid_dag_with_downstream_consumer(self):
        x = st.as_data_op(make_frame())
        t = x.drop(columns=["c"]).drop(columns=["b"]).rename(columns={"a": "z"})

        dag, *_ = optimize(t)
        self.assertEqual(len(dag), 3)                       # source + drop + rename
        drops = [op for op in dag if isinstance(op, DropOp)]
        self.assertEqual(len(drops), 1)
        result = run_plan(dag)
        self.assertEqual(list(result.columns), ["z", "d"])

    def test_fused_drop_executes_on_polars(self):
        """polars' DataFrame.drop() takes no `columns=` kwarg, so a polars-authored
        drop chain uses positional args (a shape the guard rejects) — but a fused
        DropOp produced from a pandas-authored pipeline must still execute on the
        polars backend, where process() folds kwargs['columns'] into the call."""
        fused = DropOp(args=(), kwargs={"columns": ["c", "b"]})
        frame = pl.DataFrame({"a": [1, 2], "b": [3.5, 4.5], "c": ["x", "y"]})
        FLAGS.force_polars = True
        try:
            result = fused.process("fit_transform", [frame])
        finally:
            FLAGS.force_polars = False
        self.assertEqual(result.columns, ["a"])
        self.assertEqual(result["a"].to_list(), [1, 2])


if __name__ == "__main__":
    unittest.main()
