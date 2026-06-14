"""
Pytest verification suite for the self-attention reference implementation.

Tests fall into four categories:
  1. Structural — properties that must hold for any valid input (sums, signs).
  2. Hand-computable — tiny 2-token, d_k=2 case, verified with a calculator.
  3. Cross-checks — compare against scipy.special.softmax.
  4. Numerical stability — max-subtraction trick vs naive softmax on safe inputs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from scipy.special import softmax as scipy_softmax
from reference import (
    EMBEDDINGS, W_Q, W_K, W_V, W_O,
    D_K, D_V, D_MODEL, H,
    softmax, scaled_dot_product_attention, multi_head_attention,
    causal_scaled_dot_product_attention,
)

TOL = 1e-6


# ===========================================================================
# 1. Structural checks
# ===========================================================================

class TestSoftmax:
    def test_rows_sum_to_one(self):
        x = np.array([[1.0, 2.0, 3.0],
                       [0.5, 1.5, 2.5],
                       [-1.0, 0.0, 1.0]])
        out = softmax(x, axis=-1)
        assert np.allclose(out.sum(axis=-1), 1.0, atol=TOL)

    def test_all_non_negative(self):
        x = np.random.default_rng(0).standard_normal((5, 5))
        out = softmax(x, axis=-1)
        assert np.all(out >= 0)

    def test_monotone_in_logits(self):
        # Larger logit → larger softmax value (within a row)
        x = np.array([[1.0, 3.0, 2.0]])
        out = softmax(x, axis=-1)
        assert out[0, 1] > out[0, 2] > out[0, 0]


class TestAttentionWeightsStructural:
    def setup_method(self):
        Q = EMBEDDINGS @ W_Q[0]
        K = EMBEDDINGS @ W_K[0]
        V = EMBEDDINGS @ W_V[0]
        self.result = scaled_dot_product_attention(Q, K, V)

    def test_weights_sum_to_one_per_row(self):
        assert np.allclose(self.result["weights"].sum(axis=-1), 1.0, atol=TOL)

    def test_weights_non_negative(self):
        assert np.all(self.result["weights"] >= 0)

    def test_output_shape(self):
        assert self.result["output"].shape == (3, D_V)

    def test_weights_shape(self):
        assert self.result["weights"].shape == (3, 3)

    def test_scores_shape(self):
        assert self.result["scores"].shape == (3, 3)


class TestMultiHeadAttentionStructural:
    def setup_method(self):
        self.result = multi_head_attention(EMBEDDINGS, W_Q, W_K, W_V, W_O)

    def test_output_shape(self):
        assert self.result["output"].shape == (3, D_MODEL)

    def test_concat_shape(self):
        assert self.result["concat"].shape == (3, H * D_V)

    def test_number_of_heads(self):
        assert len(self.result["heads"]) == H

    def test_each_head_weights_sum_to_one(self):
        for head in self.result["heads"]:
            assert np.allclose(head["weights"].sum(axis=-1), 1.0, atol=TOL)


# ===========================================================================
# 2. Hand-computable case
#
# 2 tokens, d_k = 2, d_v = 2.
# Q = K = [[1, 0], [0, 1]]   V = [[2, 0], [0, 2]]
#
# scores = Q K^T = [[1,0],[0,1]]
# scaled = scores / sqrt(2) = [[1/√2, 0], [0, 1/√2]]
#
# softmax row 0: exp(1/√2) / (exp(1/√2) + exp(0))
#              = e^0.7071 / (e^0.7071 + 1)
#              ≈ 2.0281 / 3.0281 ≈ 0.66966
# softmax row 1: by symmetry, same as row 0 transposed → [exp(0)/(exp(0)+exp(1/√2)), exp(1/√2)/(exp(0)+exp(1/√2))]
#              ≈ [0.33034, 0.66966]
#
# Computed by hand and with a calculator.
# ===========================================================================

class TestHandComputable:
    def setup_method(self):
        self.Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.K = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.V = np.array([[2.0, 0.0], [0.0, 2.0]])
        self.result = scaled_dot_product_attention(self.Q, self.K, self.V)

    def test_scaled_scores(self):
        expected = np.array([[1/np.sqrt(2), 0.0],
                              [0.0,          1/np.sqrt(2)]])
        assert np.allclose(self.result["scaled"], expected, atol=TOL)

    def test_attention_weights_row0(self):
        s = 1 / np.sqrt(2)
        denom = np.exp(s) + np.exp(0.0)
        w00 = np.exp(s) / denom
        w01 = np.exp(0.0) / denom
        assert np.allclose(self.result["weights"][0], [w00, w01], atol=TOL)

    def test_attention_weights_row1(self):
        s = 1 / np.sqrt(2)
        denom = np.exp(0.0) + np.exp(s)
        w10 = np.exp(0.0) / denom
        w11 = np.exp(s) / denom
        assert np.allclose(self.result["weights"][1], [w10, w11], atol=TOL)

    def test_output_is_weighted_values(self):
        expected = self.result["weights"] @ self.V
        assert np.allclose(self.result["output"], expected, atol=TOL)


# ===========================================================================
# 3. Cross-checks against scipy
# ===========================================================================

class TestCrossCheckScipy:
    def test_softmax_matches_scipy(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((6, 6))
        ours   = softmax(x, axis=-1)
        theirs = scipy_softmax(x, axis=-1)
        assert np.allclose(ours, theirs, atol=TOL)

    def test_attention_weights_match_scipy_softmax(self):
        Q = EMBEDDINGS @ W_Q[0]
        K = EMBEDDINGS @ W_K[0]
        d_k = Q.shape[-1]
        scaled = (Q @ K.T) / np.sqrt(d_k)
        ours   = softmax(scaled, axis=-1)
        theirs = scipy_softmax(scaled, axis=-1)
        assert np.allclose(ours, theirs, atol=TOL)


# ===========================================================================
# 4. Numerical stability
# ===========================================================================

class TestNumericalStability:
    def test_large_logits_no_overflow(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        out = softmax(x, axis=-1)
        assert np.all(np.isfinite(out))
        assert np.allclose(out.sum(axis=-1), 1.0, atol=TOL)

    def test_negative_logits(self):
        x = np.array([[-1000.0, -1001.0, -999.0]])
        out = softmax(x, axis=-1)
        assert np.all(np.isfinite(out))
        assert np.allclose(out.sum(axis=-1), 1.0, atol=TOL)

    def test_max_subtraction_same_as_naive_on_safe_input(self):
        x = np.array([[0.5, 1.0, 1.5, 0.0]])
        stable = softmax(x, axis=-1)
        naive = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
        assert np.allclose(stable, naive, atol=TOL)


# ===========================================================================
# 5. Causal (decoder) attention mask
#
# Structural properties that must hold for any causal attention:
#   - weight[i, j] ≈ 0 for all j > i  (no future information)
#   - each row still sums to 1          (valid probability distribution)
#   - weight[0, 0] == 1.0              (first token sees only itself)
#   - all weights in last row > 0      (last token sees all prior tokens)
#   - masked score values (upper triangle) == mask_value (-1e9)
# ===========================================================================

class TestCausalMask:
    def setup_method(self):
        Q = EMBEDDINGS @ W_Q[0]
        K = EMBEDDINGS @ W_K[0]
        V = EMBEDDINGS @ W_V[0]
        self.result = causal_scaled_dot_product_attention(Q, K, V)
        self.n = self.result["weights"].shape[0]

    def test_future_weights_are_zero(self):
        """Position i must not attend to any j > i."""
        w = self.result["weights"]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                assert w[i, j] < TOL, \
                    f"Causal mask failed: weights[{i},{j}] = {w[i,j]:.2e}"

    def test_rows_still_sum_to_one(self):
        """Masking changes the logits but softmax must still produce a distribution."""
        assert np.allclose(self.result["weights"].sum(axis=-1), 1.0, atol=TOL)

    def test_first_token_attends_only_to_itself(self):
        """Row 0 has all future positions masked; weight[0,0] must equal 1.0."""
        assert np.allclose(self.result["weights"][0, 0], 1.0, atol=TOL)

    def test_weights_non_negative(self):
        assert np.all(self.result["weights"] >= 0)

    def test_last_row_has_access_to_all_tokens(self):
        """The last position can attend to every token, so all weights > 0."""
        last = self.result["weights"][-1, :]
        assert np.all(last > TOL), f"Last row has near-zero weights: {last}"

    def test_masked_positions_hold_mask_value(self):
        """Upper triangle of the masked scores must equal the mask_value."""
        masked = self.result["masked"]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                assert masked[i, j] == pytest.approx(-1e9), \
                    f"masked[{i},{j}] should be -1e9, got {masked[i,j]}"

    def test_unmasked_positions_unchanged(self):
        """Lower triangle of masked must equal lower triangle of scaled."""
        scaled = self.result["scaled"]
        masked = self.result["masked"]
        for i in range(self.n):
            for j in range(i + 1):
                assert masked[i, j] == pytest.approx(scaled[i, j], abs=TOL), \
                    f"Unmasked position [{i},{j}] was altered"
