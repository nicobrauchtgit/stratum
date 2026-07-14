import unittest
import stratum as st
import numpy as np
import scipy.special as sp
from stratum.optimizer._optimize import optimize, OptConfig
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer.ir._ops import ValueOp


def _run_plan(dag):
    """Execute a linearized plan and return the root op's value."""
    cache = {}
    for op in dag:
        cache[id(op)] = op.process("fit", [cache[id(i)] for i in op.inputs])
    return cache[id(dag[-1])]


def _has_softmax(dag):
    """True if any op in the plan is the fused GENERIC softmax op."""
    return any(isinstance(op, NumericOp)
               and op.type is NumericOpType.GENERIC
               and op.func is sp.softmax
               for op in dag)


class TestNumericComplexFanout(unittest.TestCase):

    def _opt(self, dag):
        return optimize(dag, OptConfig(
            algebraic_rewrite_config=AlgebraicRewritesConfig(constant_folding=False),
        ))

    def test_fanout_before_and_after_chain(self):
        """
        Scenario:
          a -> [log, d]
          log -> [exp]
          exp -> [b, c]
        
        Expected after optimization:
          a -> [d, b, c]
        """
        a = st.as_data_op(1.0)
        
        # Branch 1: log -> exp -> [b, c]
        log_a = a.skb.apply_func(np.log)
        exp_log_a = log_a.skb.apply_func(np.exp)
        b = exp_log_a + 1.0
        c = exp_log_a + 2.0
        
        # Branch 2: d
        d = a + 10.0
        
        # Root combining all branches
        t1 = d + b
        final = t1 + c
        
        linearized_dag, *_ = self._opt(final)
        
        # 1. Find the original 'a' Op
        a_op = linearized_dag[0]
        self.assertIsInstance(a_op, ValueOp)
        self.assertEqual(a_op.value, 1.0)
        
        # 2. Verify 'a' has 3 outputs: d, b, and c
        self.assertEqual(len(a_op.outputs), 3, f"Expected 3 outputs for 'a', found {len(a_op.outputs)}")
        
        # All outputs of 'a' should be NumericOp(ADD) now (BinOp(+) is extracted to NumericOp)
        for out in a_op.outputs:
            self.assertIsInstance(out, NumericOp)
            self.assertEqual(out.type, NumericOpType.ADD)
            self.assertIn(a_op, out.inputs)

        # 3. Check that no log or exp ops are left
        for op in linearized_dag:
            if isinstance(op, NumericOp):
                self.assertNotIn(op.type, [NumericOpType.LOG, NumericOpType.EXP])

    def test_fanout_on_op2_only(self):
        """
        Scenario:
          a -> log -> exp -> [b, c]

        Expected after optimization:
          a -> [b, c]
        """
        a = st.as_data_op(1.0)
        log_a = a.skb.apply_func(np.log)
        exp_log_a = log_a.skb.apply_func(np.exp)
        b = exp_log_a + 1.0
        c = exp_log_a + 2.0
        final = b + c

        linearized_dag, *_ = self._opt(final)

        a_op = linearized_dag[0]
        self.assertIsInstance(a_op, ValueOp)
        self.assertEqual(a_op.value, 1.0)

        self.assertEqual(len(a_op.outputs), 2)
        for out in a_op.outputs:
            self.assertIsInstance(out, NumericOp)
            self.assertEqual(out.type, NumericOpType.ADD)

        for op in linearized_dag:
            if isinstance(op, NumericOp):
                self.assertNotIn(op.type, [NumericOpType.LOG, NumericOpType.EXP])

    def test_chain_is_root(self):
        """
        Scenario:
          a -> log -> exp  (exp is the root)

        Expected after optimization:
          root is a
        """
        a = st.as_data_op(1.0)
        log_a = a.skb.apply_func(np.log)
        exp_log_a = log_a.skb.apply_func(np.exp)

        linearized_dag, *_ = self._opt(exp_log_a)

        a_op = linearized_dag[0]
        self.assertIsInstance(a_op, ValueOp)
        self.assertEqual(a_op.value, 1.0)

        self.assertEqual(len(a_op.outputs), 0)
        self.assertIs(linearized_dag[-1], a_op)

        for op in linearized_dag:
            if isinstance(op, NumericOp):
                self.assertNotIn(op.type, [NumericOpType.LOG, NumericOpType.EXP])

    def test_chain_is_root_with_other_fanout(self):
        """
        Scenario:
          a -> [log, d]
          log -> exp -> BinOp (root)
        
        Expected after optimization:
          a -> [d, combined]
          d -> combined  (combined = a + d is the root)
        """
        a = st.as_data_op(1.0)
        log_a = a.skb.apply_func(np.log)
        exp_log_a = log_a.skb.apply_func(np.exp)
        
        # Add another branch so 'a' has fan-out
        d = a + 10.0
        
        combined = exp_log_a + d
        linearized_dag, *_ = self._opt(combined)
        
        a_op = linearized_dag[0]
        self.assertIsInstance(a_op, ValueOp)
        self.assertEqual(a_op.value, 1.0)

        # 'a' should now connect directly to the root (the BinOp from combined)
        # and to 'd'.
        self.assertEqual(len(a_op.outputs), 2)

        # Verify no NumericOps remain
        for op in linearized_dag:
            if isinstance(op, NumericOp):
                self.assertNotIn(op.type, [NumericOpType.LOG, NumericOpType.EXP])

    def test_softmax_fires(self):
        x = st.as_data_op(np.array([1., 2., 3.]))
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)
        soft = e / s

        linearized_dag, *_ = optimize(soft)
        # Expected shape: ValueOp(x) + NumericOp(GENERIC, sp.softmax)
        self.assertEqual(len(linearized_dag), 2)
        softmax_op = linearized_dag[1]
        self.assertIsInstance(softmax_op, NumericOp)
        self.assertEqual(softmax_op.type, NumericOpType.GENERIC)
        self.assertIs(softmax_op.func, sp.softmax)

        # Equivalence: the fused plan must produce the same values as the
        # unoptimized exp(x)/sum(exp(x)) graph (not merely equal sp.softmax to
        # itself, which the fused op trivially would).
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(softmax=False),
        )
        unfused_dag, *_ = optimize(soft, config=config)
        np.testing.assert_allclose(_run_plan(linearized_dag), _run_plan(unfused_dag))

    def test_softmax_disabled(self):
        x = st.as_data_op(np.array([1., 2., 3.]))
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)
        soft = e / s

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(softmax=False),
        )
        linearized_dag, *_ = optimize(soft, config=config)
        # ValueOp + EXP + SUM + DIV = 4 ops
        self.assertEqual(len(linearized_dag), 4)

    def test_no_rewrite_when_exp_has_third_consumer(self):
        """EXP feeds softmax pattern PLUS another consumer — must NOT rewrite,
        otherwise the third consumer loses its input."""
        x = st.as_data_op(np.array([1., 2., 3.]))
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)
        soft = e / s
        other = e + 1.0            # third consumer of e
        root = soft + other

        linearized_dag, *_ = optimize(root)
        # No softmax collapse — verify by checking no GENERIC op with func=sp.softmax
        self.assertFalse(_has_softmax(linearized_dag))

    def test_no_rewrite_reversed_divide(self):
        """sum(exp(x)) / exp(x) is 1/softmax(x), not softmax(x)."""
        x = st.as_data_op(np.array([1., 2., 3.]))
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)
        inv = s / e     # reversed order

        linearized_dag, *_ = optimize(inv)
        self.assertFalse(_has_softmax(linearized_dag))

    def test_no_rewrite_when_sum_has_axis(self):
        """sum(exp(x), axis=0) must NOT rewrite — axis-aware softmax not supported yet."""
        x = st.as_data_op(np.array([[1., 2., 3.], [4., 5., 6.]]))
        e = x.skb.apply_func(np.exp)
        # apply_func passes extra kwargs through to the func
        s = e.skb.apply_func(np.sum, axis=0)
        soft = e / s

        linearized_dag, *_ = optimize(soft)
        self.assertFalse(_has_softmax(linearized_dag))

    def test_softmax_fires_mid_dag(self):
        """Fusion with a consumer AFTER the divide — exercises the
        downstream.replace_input(...) path of the action, which the root-position
        tests never reach."""
        x = st.as_data_op(np.array([1., 2., 3.]))
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)
        soft = e / s
        t = soft * 2.0                       # downstream consumer of the fused op

        linearized_dag, *_ = optimize(t)
        # ValueOp + softmax + MULTIPLY
        self.assertEqual(len(linearized_dag), 3)
        softmax_op = linearized_dag[1]
        self.assertIs(softmax_op.func, sp.softmax)
        mul_op = linearized_dag[2]
        self.assertEqual(mul_op.type, NumericOpType.MULTIPLY)
        self.assertIn(softmax_op, mul_op.inputs)

    def test_softmax_fires_on_cse_merged_exps(self):
        """exp(x) written twice as distinct expressions: CSE (on by default) merges
        the two CallOps, then the identity check holds and the fusion fires."""
        x = st.as_data_op(np.array([1., 2., 3.]))
        e1 = x.skb.apply_func(np.exp)
        e2 = x.skb.apply_func(np.exp)        # distinct DataOp, structurally equal
        s = e2.skb.apply_func(np.sum)
        soft = e1 / s

        linearized_dag, *_ = optimize(soft)
        self.assertEqual(len(linearized_dag), 2)
        self.assertIs(linearized_dag[1].func, sp.softmax)

    def test_no_fuse_distinct_exps_when_cse_disabled(self):
        """Without CSE the two exp expressions stay distinct ops, the identity
        check rejects, and no fusion happens (correct: equality is unproven).
        NOTE: optimize() gates CSE on the global FLAGS.cse (_optimize.py:95);
        OptConfig.cse exists but is dead — hence the FLAGS toggle here."""
        from stratum._config import FLAGS
        x = st.as_data_op(np.array([1., 2., 3.]))
        e1 = x.skb.apply_func(np.exp)
        e2 = x.skb.apply_func(np.exp)
        s = e2.skb.apply_func(np.sum)
        soft = e1 / s

        FLAGS.cse = False
        try:
            linearized_dag, *_ = optimize(soft)
        finally:
            FLAGS.cse = True
        self.assertFalse(_has_softmax(linearized_dag))
        self.assertEqual(len(linearized_dag), 5)   # ValueOp + 2×EXP + SUM + DIV

    def test_softmax_fires_with_nan_input(self):
        """NaN-sensitive semantics: the fusion must still fire on NaN-containing
        input, and it must not change the result versus the unfused
        exp(x)/sum(exp(x)) graph (NaN propagation included)."""
        data = np.array([1.0, np.nan, 3.0])
        x = st.as_data_op(data)
        e = x.skb.apply_func(np.exp)
        s = e.skb.apply_func(np.sum)
        soft = e / s

        linearized_dag, *_ = optimize(soft)
        self.assertEqual(len(linearized_dag), 2)
        self.assertIs(linearized_dag[1].func, sp.softmax)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(softmax=False),
        )
        unfused_dag, *_ = optimize(soft, config=config)
        np.testing.assert_allclose(
            _run_plan(linearized_dag), _run_plan(unfused_dag), equal_nan=True)

if __name__ == "__main__":
    unittest.main()
