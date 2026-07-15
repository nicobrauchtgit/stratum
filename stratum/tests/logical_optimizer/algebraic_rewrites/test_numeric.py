import unittest
import stratum as st
import numpy as np
from stratum.optimizer._optimize import  optimize
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer.ir._ops import OperandRef

class TestCSE(unittest.TestCase):

    def test_log_exp1(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_log_exp2(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)
        t3 = t2.skb.apply_func(np.log1p)

        out, *_ = optimize(t3)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].value, 1)

    def test_exp_log1(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.log)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_exp_log2(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.log)
        t3 = t2.skb.apply_func(np.log1p)

        out, *_ = optimize(t3)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].value, 1)

    def test_log1p_expm1(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log1p)
        t2 = t1.skb.apply_func(np.expm1)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_expm1_log1p(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.expm1)
        t2 = t1.skb.apply_func(np.log1p)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_log_log1p(self):
        "no algebraic rewrite should be applied here "
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.log1p)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)

    def test_expm1_log1p_disabled(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.expm1)
        t2 = t1.skb.apply_func(np.log1p)
        config = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig(log1p_expm1 = False,expm1_log1p = False))
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].value, 1)

    def test_log_log1p_exp(self):
        "no algebraic rewrite should be applied here "
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.log1p)
        t3 = t2.skb.apply_func(np.exp)
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 4)

    def test_log1p_log1p_exp(self):
        "no algebraic rewrite should be applied here "
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log1p)
        t2 = t1.skb.apply_func(np.log1p)
        t3 = t2.skb.apply_func(np.exp)
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 4)

    def test_disable_log_exp_rewrite1(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(log_exp=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_disable_log_exp_rewrite2(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(exp_log=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)

    def test_sqrt_square_via_np_square(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.sqrt)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)
        # abs(4) = 4
        self.assertEqual(out[0].value, 4)

    def test_sqrt_pow2(self):
        df = st.as_data_op(-3)
        t1 = df ** 2
        t2 = t1.skb.apply_func(np.sqrt)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)
        # abs(-3) = 3
        self.assertEqual(out[0].value, -3)

    def test_sqrt_square_with_trailing_op(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.sqrt)
        t3 = t2.skb.apply_func(np.log1p)

        out, *_ = optimize(t3)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].value, 4)

    def test_disable_sqrt_square_rewrite(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.sqrt)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(sqrt_square=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_no_rewrite_sqrt_only(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.sqrt)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)

    def test_sqrt_square_produces_abs_op(self):
        """Rewrite must insert an abs op, not identity — critical for negative inputs."""
        df = st.as_data_op(-5)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.sqrt)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.ABS)

    def test_sqrt_pow2_produces_abs_op(self):
        """BinOp(**2) → sqrt rewrite must also produce abs, not identity."""
        df = st.as_data_op(-5)
        t1 = df ** 2
        t2 = t1.skb.apply_func(np.sqrt)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.ABS)

    def test_no_rewrite_square_log(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.square)
        t2 = t1.skb.apply_func(np.log)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)

    def test_no_rewrite_sqrt_exp(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.sqrt)
        t2 = t1.skb.apply_func(np.exp)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)

    def test_no_rewrite_pow3_sqrt(self):
        """x**3 → sqrt should not rewrite; only x**2 qualifies."""
        df = st.as_data_op(4)
        t1 = df ** 3
        t2 = t1.skb.apply_func(np.sqrt)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)

    def test_disable_sqrt_square_does_not_affect_log_exp(self):
        """Disabling sqrt_square must not suppress other algebraic rewrites."""
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(sqrt_square=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)

    def test_eliminate_identity_operation(self):
        df = st.as_data_op(2)
        t1 = df * 1
        t2 = t1 + 3

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 5)

    def test_disable_eliminate_identity_operation(self):
        df = st.as_data_op(2)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(identity_op=False),
        )
        t1 = df * 1
        t2 = t1 + 3

        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        multiply_result = out[1].process("fit", [out[0].value])
        self.assertEqual(multiply_result, 2)
        self.assertEqual(out[2].process("fit", [multiply_result]), 5)

    def test_eliminate_identity_operation_root_safe(self):
        value = st.as_data_op(2)
        root = value * 1

        out, *_ = optimize(root)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].process("fit", [out[0].value]), 2)

    def test_eliminate_identity_operation_dedups_repeated_input(self):
        value = st.as_data_op(2)
        a = value + (value * 1)
        b = (value * 1) + value
        root = a + b

        out, *_ = optimize(root)

        self.assertEqual(len(out), 4)
        self.assertEqual(out[1].inputs, [out[0]])
        self.assertEqual(out[2].inputs, [out[0]])
        self.assertEqual(out[2].opt_operand, OperandRef(0))
        left_sum = out[1].process("fit", [out[0].value])
        right_sum = out[2].process("fit", [out[0].value])
        self.assertEqual(out[3].process("fit", [left_sum, right_sum]), 8)

    def test_abs_abs_collapses_to_single_abs(self):
        df = st.as_data_op(-3)
        t1 = df.skb.apply_func(np.abs)
        t2 = t1.skb.apply_func(np.abs)
        out, *_ = optimize(t2)

        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.ABS)

    def test_single_abs_untouched(self):
        df = st.as_data_op(-3)
        t1 = df.skb.apply_func(np.abs)
        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)

    def test_abs_abs_with_trailing_op(self):
        df = st.as_data_op(-3)
        t1 = df.skb.apply_func(np.abs)
        t2 = t1.skb.apply_func(np.abs)
        t3 = t2.skb.apply_func(np.log1p)
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 3)

    def test_no_rewrite_abs_sqrt(self):
        df = st.as_data_op(4)
        t1 = df.skb.apply_func(np.abs)
        t2 = t1.skb.apply_func(np.sqrt)
        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)

    def test_abs_abs_disabled(self):
        df = st.as_data_op(-3)
        t1 = df.skb.apply_func(np.abs)
        t2 = t1.skb.apply_func(np.abs)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(abs_abs=False)
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)

    def test_disable_abs_abs_does_not_affect_log_exp(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(abs_abs=False)
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)

    def test_eliminate_add_zero(self):
        df = st.as_data_op(2)
        t1 = df + 0
        t2 = t1 + 3
        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 5)

    def test_eliminate_zero_add(self):
        df = st.as_data_op(2)
        t1 = 0 + df
        t2 = t1 + 3
        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 5)

    def test_eliminate_add_zero_root_safe(self):
        df = st.as_data_op(2)
        root = df + 0
        out, *_ = optimize(root)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].process("fit", [out[0].value]), 2)

    def test_disable_eliminate_add_zero(self):
        df = st.as_data_op(2)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(add_zero=False),
        )
        t1 = 0 + df
        t2 = t1 + 3
        out, *_ = optimize(t2, config=config)
        print(out)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[1].process("fit", [out[0].value]), 2)

    def test_no_rewrite_add_nonzero(self):
        df = st.as_data_op(2)
        t1 = df + 1
        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 3)

    def test_add_zero_with_trailing_op(self):
        df = st.as_data_op(2)
        t1 = df + 0
        t2 = t1 + 3
        t3 = t2.skb.apply_func(np.log1p)
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 3)

    def test_add_zero_and_identity_operation(self):
        df = st.as_data_op(2)
        t1 = df * 1
        t2 = t1 + 0
        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 2)

    def test_exp_minus_one(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 - 1
        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)                                  # source + expm1 (was 3)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXPM1)
        self.assertEqual(out[1].process("fit", [out[0].value]), 0)     # expm1(0) == 0

    def test_no_rewrite_one_minus_exp(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = 1 - t1                                                     # reversed: NOT expm1
        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXP)
        self.assertEqual(out[1].process("fit", [out[0].value]), 1)

    def test_no_rewrite_exp_minus_two(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 - 2                                                     # constant != 1
        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXP)

    def test_disable_exp_minus_one(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 - 1
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(exp_minus_one=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXP)

    def test_exp_minus_one_and_identity_operation(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 + 0
        t3 = t2 - 1
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.EXPM1)

    def test_log1p_of_exp_minus_one_reduces_to_input(self):
        df = st.as_data_op(0)
        t1 = df.skb.apply_func(np.exp)
        t2 = t1 - 1
        t3 = t2.skb.apply_func(np.log1p)  # log1p(exp(df)-1)
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 1)  # -> log1p(expm1(df)) -> df
        self.assertEqual(out[0].value, 0)

    def test_exp_log_minus_one_not_fused(self):
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)
        t3 = t2 - 1  # exp(log(x)) - 1
        out, *_ = optimize(t3)
        self.assertEqual(len(out), 2)  # exp/log cancel -> x - 1
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.SUBTRACT)

    def test_eliminate_identity_subtract(self):
        """x - 0  →  x"""
        df = st.as_data_op(5)
        t1 = df - 0
        t2 = t1 + 3

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), 8)

    def test_eliminate_identity_subtract_root_safe(self):
        """When x - 0 is the root, the rewrite must not break the DAG."""
        value = st.as_data_op(7)
        root = value - 0

        out, *_ = optimize(root)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].process("fit", [out[0].value]), 7)

    def test_disable_eliminate_identity_subtract(self):
        """Disabling identity_subtract must leave x - 0 untouched."""
        df = st.as_data_op(5)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(identity_subtract=False),
        )
        t1 = df - 0
        t2 = t1 + 3

        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 3)
        subtract_result = out[1].process("fit", [out[0].value])
        self.assertEqual(subtract_result, 5)
        self.assertEqual(out[2].process("fit", [subtract_result]), 8)

    def test_no_rewrite_const_minus_var(self):
        """0 - x  should NOT be rewritten (it is not an identity)."""
        df = st.as_data_op(5)
        t1 = 0 - df
        t2 = t1 + 3

        out, *_ = optimize(t2)
        # x - 0 would collapse to 2 ops; 0 - x stays as 3 ops
        self.assertEqual(len(out), 3)

    def test_eliminate_div_by_one_fires(self):
        df = st.as_data_op(6)
        t1 = df / 1

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 1)                        # only the ValueOp remains
        self.assertEqual(out[0].value, 6)

    def test_eliminate_div_by_one_in_chain(self):
        df = st.as_data_op(6)
        t1 = df / 1
        t2 = t1 + 3

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 2)                        # ValueOp + ADD
        self.assertEqual(out[1].process("fit", [out[0].value]), 9)

    def test_eliminate_div_by_one_disabled(self):
        df = st.as_data_op(6)
        t1 = df / 1
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(div_by_one=False),
        )
        out, *_ = optimize(t1, config=config)
        self.assertEqual(len(out), 2)                        # DIV remains

    def test_no_rewrite_one_over_x(self):
        """1 / x must NOT be rewritten — DIVIDE is non-commutative."""
        df = st.as_data_op(6)
        t1 = 1 / df

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)                        # ValueOp + DIV(reversed=True)
        # sanity: check the DIV is still there and reversed
        div_op = out[1]
        self.assertIsInstance(div_op, NumericOp)
        self.assertEqual(div_op.type, NumericOpType.DIVIDE)
        self.assertTrue(div_op.reversed)

    def test_no_rewrite_div_by_other_constant(self):
        """x / 2 must NOT be rewritten — only constant 1 counts."""
        df = st.as_data_op(6)
        t1 = df / 2

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)

    def test_no_crash_div_by_ndarray_constant(self):
        """df / ndarray must neither crash nor rewrite (ambiguous-truth-value trap)."""
        df = st.as_data_op(np.array([6.0, 8.0]))
        t1 = df / np.array([1.0, 1.0])

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)                        # ValueOp + DIV survive

    def test_no_crash_mul_by_ndarray_constant(self):
        """Regression for the pre-existing crash in mul-by-one (#93): before the
        isinstance guard, `df * np.array([...])` raised 'truth value of an array
        is ambiguous' inside match_identity_operation."""
        df = st.as_data_op(np.array([6.0, 8.0]))
        t1 = df * np.array([2.0, 3.0])

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)                        # ValueOp + MUL survive

    def test_neg_neg_via_np_negative_apply_func(self):
        df = st.as_data_op(5)
        t1 = df.skb.apply_func(np.negative)
        t2 = t1.skb.apply_func(np.negative)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 5)

    def test_neg_neg_end_to_end_negative_input(self):
        """Numerical: -(-(-3)) trap — but we only collapse pairs, so (-(-3)) == 3."""
        df = st.as_data_op(-3)
        t1 = df.skb.apply_func(np.negative)
        t2 = t1.skb.apply_func(np.negative)

        out, *_ = optimize(t2)
        self.assertEqual(out[0].value, -3)

    def test_neg_neg_disabled(self):
        df = st.as_data_op(5)
        t1 = df.skb.apply_func(np.negative)
        t2 = t1.skb.apply_func(np.negative)
        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(neg_neg=False),
        )
        out, *_ = optimize(t2, config=config)
        # ValueOp + 2 NumericOp(GENERIC, np.negative)
        self.assertEqual(len(out), 3)

    def test_no_rewrite_neg_alone(self):
        """Single np.negative must not be rewritten."""
        df = st.as_data_op(5)
        t1 = df.skb.apply_func(np.negative)

        out, *_ = optimize(t1)
        self.assertEqual(len(out), 2)

    def test_no_rewrite_neg_log(self):
        """Different funcs in the chain must not match."""
        df = st.as_data_op(5)
        t1 = df.skb.apply_func(np.negative)
        t2 = t1.skb.apply_func(np.log)

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)

    def test_neg_neg_with_trailing_op(self):
        """Chain in the middle of a bigger DAG (root-safety of the reused helper)."""
        df = st.as_data_op(5)
        t1 = df.skb.apply_func(np.negative)
        t2 = t1.skb.apply_func(np.negative)
        t3 = t2.skb.apply_func(np.log)

        out, *_ = optimize(t3)
        # ValueOp + LOG (neg-neg collapsed)
        self.assertEqual(len(out), 2)

    def test_neg_neg_triple_chain_collapses_to_one(self):
        """Odd-length chain: neg(neg(neg(x))) -> neg(x). Collapsing pairs from the
        bottom up must leave exactly one negation and must not corrupt the DAG
        (a naive top-down match re-processes a detached op and crashes here)."""
        df = st.as_data_op(7)
        t = df.skb.apply_func(np.negative)
        t = t.skb.apply_func(np.negative)
        t = t.skb.apply_func(np.negative)

        out, *_ = optimize(t)
        self.assertEqual(len(out), 2)                # ValueOp + one NumericOp(neg)
        self.assertEqual(out[1].process("fit", [out[0].value]), -7)

    def test_neg_neg_quad_chain_collapses_to_zero(self):
        """Even-length chain: four negations cancel completely."""
        df = st.as_data_op(7)
        t = df.skb.apply_func(np.negative)
        for _ in range(3):
            t = t.skb.apply_func(np.negative)

        out, *_ = optimize(t)
        self.assertEqual(len(out), 1)                # ValueOp only
        self.assertEqual(out[0].value, 7)

    def test_neg_neg_five_chain_end_to_end(self):
        """Longer odd chain: five negations reduce to one; value stays negated."""
        df = st.as_data_op(3)
        t = df.skb.apply_func(np.negative)
        for _ in range(4):
            t = t.skb.apply_func(np.negative)

        out, *_ = optimize(t)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].process("fit", [out[0].value]), -3)

    def test_direct_ufunc_call_not_matched(self):
        """np.negative(np.negative(df)) converts to UnaryOp (not CallOp), so it
        never extracts to GENERIC NumericOps and this rewrite must not touch it.
        Pins the current boundary; the UnaryOp-canonicalization follow-up will
        collapse this shape too."""
        df = st.as_data_op(5)
        t2 = np.negative(np.negative(df))

        out, *_ = optimize(t2)
        self.assertEqual(len(out), 3)                # ValueOp + 2 UnaryOps survive

    def test_neg_neg_nan_equivalence(self):
        """neg(neg(nan)) must equal unoptimized result — identity holds for NaN."""
        import math
        df = st.as_data_op(float('nan'))
        t1 = df.skb.apply_func(np.negative)
        t2 = t1.skb.apply_func(np.negative)
        out_opt, *_ = optimize(t2)
        self.assertEqual(len(out_opt), 1)
        self.assertTrue(math.isnan(out_opt[0].value))

    def test_neg_neg_array_with_nan(self):
        """array containing NaN: rewrite fires and NaN is preserved."""
        arr = np.array([1.0, float('nan'), -3.0])
        df = st.as_data_op(arr)
        t1 = df.skb.apply_func(np.negative)
        t2 = t1.skb.apply_func(np.negative)
        out_opt, *_ = optimize(t2)
        self.assertEqual(len(out_opt), 1)
        np.testing.assert_array_equal(out_opt[0].value, arr)
    # --- log-sum-exp -----------------------------------------------------

    def test_log_sum_exp_rewrite(self):
        """log(sum(exp(x))) → logsumexp(x)"""
        df = st.as_data_op(np.array([1.0, 2.0, 3.0]))
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.sum)
        t3 = t2.skb.apply_func(np.log)

        out, *_ = optimize(t3)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], NumericOp)
        self.assertEqual(out[1].type, NumericOpType.GENERIC)
        result = out[1].process("fit", [out[0].value])
        expected = np.log(np.sum(np.exp([1.0, 2.0, 3.0])))
        np.testing.assert_almost_equal(result, expected)

    def test_log_sum_exp_rewrite_root_safe(self):
        """When log(sum(exp(x))) is the root, DAG must not break."""
        df = st.as_data_op(np.array([0.0, 1.0]))
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.sum)
        root = t2.skb.apply_func(np.log)

        out, *_ = optimize(root)
        self.assertEqual(len(out), 2)
        result = out[1].process("fit", [out[0].value])
        expected = np.log(np.sum(np.exp([0.0, 1.0])))
        np.testing.assert_almost_equal(result, expected)

    def test_log_sum_exp_with_trailing_op(self):
        """log(sum(exp(x))) + 1  →  logsumexp(x) + 1"""
        df = st.as_data_op(np.array([1.0, 2.0, 3.0]))
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.sum)
        t3 = t2.skb.apply_func(np.log)
        t4 = t3 + 1

        out, *_ = optimize(t4)
        self.assertEqual(len(out), 3)

    def test_log_sum_exp_disabled(self):
        """Disabling log_sum_exp must leave the 3-op chain untouched."""
        df = st.as_data_op(np.array([1.0, 2.0, 3.0]))
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.sum)
        t3 = t2.skb.apply_func(np.log)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(log_sum_exp=False),
        )
        out, *_ = optimize(t3, config=config)
        self.assertEqual(len(out), 4)

    def test_no_rewrite_sum_log_exp(self):
        """sum(log(exp(x))) must NOT match (wrong order)."""
        df = st.as_data_op(np.array([1.0, 2.0]))
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.log)
        t3 = t2.skb.apply_func(np.sum)

        out, *_ = optimize(t3)
        self.assertEqual(len(out), 2)

    def test_no_rewrite_log_mean_exp(self):
        """log(mean(exp(x))) must NOT match (mean, not sum)."""
        df = st.as_data_op(np.array([1.0, 2.0, 3.0]))
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.mean)
        t3 = t2.skb.apply_func(np.log)

        out, *_ = optimize(t3)
        self.assertEqual(len(out), 4)

    def test_disable_log_sum_exp_does_not_affect_log_exp(self):
        """Disabling log_sum_exp must not suppress other rewrites."""
        df = st.as_data_op(1)
        t1 = df.skb.apply_func(np.log)
        t2 = t1.skb.apply_func(np.exp)

        config = OptConfig(
            algebraic_rewrites=True,
            algebraic_rewrite_config=AlgebraicRewritesConfig(log_sum_exp=False),
        )
        out, *_ = optimize(t2, config=config)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].value, 1)

    def test_no_rewrite_exp_sum_sqrt(self):
        """exp → sum → sqrt must NOT match (third op is SQRT, not LOG)."""
        df = st.as_data_op(np.array([1.0, 2.0, 3.0]))
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.sum)
        t3 = t2.skb.apply_func(np.sqrt)

        out, *_ = optimize(t3)
        self.assertEqual(len(out), 4)

    def test_no_rewrite_sum_has_fanout(self):
        """exp → sum → [log, sqrt] must NOT match (sum has fan-out)."""
        df = st.as_data_op(np.array([1.0, 2.0, 3.0]))
        t1 = df.skb.apply_func(np.exp)
        t2 = t1.skb.apply_func(np.sum)
        t3 = t2.skb.apply_func(np.log)
        t4 = t2.skb.apply_func(np.sqrt)
        combined = t3 + t4

        out, *_ = optimize(combined)
        num_generic = sum(1 for op in out if isinstance(op, NumericOp) and op.type == NumericOpType.GENERIC)
        self.assertGreaterEqual(num_generic, 1)
