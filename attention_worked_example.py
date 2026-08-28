"""
Causal self-attention, worked by hand and checked with numpy only (no PyTorch).

This script produces every number shown on the "worked example" slides in
transformers-slides.html. Run it and compare its printed output to the slide —
they must match exactly. It is written so you can read it top to bottom and see
precisely which matrix multiplication produces which number.

Setup: 3 tokens, embedding width d = 3, single head with d_k = d_v = d = 3 — the
standard single-head case, where the attention output already has the same shape
as X and can be added directly to the residual stream (no output projection W_O
needed; see the note on the multi-head slide for why multi-head is different).
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(scores):
    """Row-wise softmax: turn each row of raw scores into a probability distribution."""
    shifted = scores - np.max(scores, axis=-1, keepdims=True)  # for numerical stability only
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. Input: X is the token-sequence matrix, one row per token.
#    X has shape (n, d) = (3 tokens, embedding width 3).
# ---------------------------------------------------------------------------
X = np.array([
    [1., 0., 1.],   # x1 — token 1's embedding
    [0., 1., 1.],   # x2 — token 2's embedding
    [1., 1., 0.],   # x3 — token 3's embedding
])
print("X (n=3, d=3) =\n", X)

# ---------------------------------------------------------------------------
# 2. Learned projection matrices. Shapes: (d, d_k) so that Q = X @ W_Q has
#    shape (n, d_k). Here d_k=d_v=d=3, so Q, K, V all stay the same width as X.
# ---------------------------------------------------------------------------
W_Q = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [1., 1., 1.],
])
W_K = np.array([
    [1., 1., 0.],
    [1., 0., 1.],
    [0., 1., 1.],
])
W_V = np.array([
    [1.,  0., 0.],
    [0.,  1., 0.],
    [1., -1., 1.],
])
print("\nW_Q (d=3, d_k=3) =\n", W_Q)
print("W_K (d=3, d_k=3) =\n", W_K)
print("W_V (d=3, d_v=3) =\n", W_V)

# ---------------------------------------------------------------------------
# 3. Q = X W_Q, K = X W_K, V = X W_V — three linear projections of the same X.
# ---------------------------------------------------------------------------
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
print("\nQ = X @ W_Q =\n", Q)
print("K = X @ W_K =\n", K)
print("V = X @ W_V =\n", V)

# ---------------------------------------------------------------------------
# 4. Raw compatibility scores: Q K^T, shape (n, n). Row i = how much token i's
#    query matches every token's key.
# ---------------------------------------------------------------------------
scores = Q @ K.T
print("\nQ @ K^T =\n", scores)

# ---------------------------------------------------------------------------
# 5. Scale by sqrt(d_k) — keeps the scores from growing with d_k so softmax
#    doesn't saturate.
# ---------------------------------------------------------------------------
d_k = W_Q.shape[1]
scaled_scores = scores / np.sqrt(d_k)
print(f"\nQ @ K^T / sqrt(d_k)   [d_k={d_k}, sqrt(d_k)={np.sqrt(d_k):.4f}] =\n", scaled_scores)

# ---------------------------------------------------------------------------
# 6. Causal mask: token i may only look at tokens j <= i. Set every j > i
#    entry to -infinity so it gets exactly zero weight after softmax.
# ---------------------------------------------------------------------------
n = X.shape[0]
causal_mask = np.triu(np.full((n, n), -np.inf), k=1)
causal_mask = np.nan_to_num(causal_mask, neginf=-1e9)  # avoid NaN from -inf + -inf
masked_scores = scaled_scores + causal_mask
print("\ncausal mask (upper triangle blocked) =\n", causal_mask)
print("scaled scores + mask =\n", masked_scores)

# ---------------------------------------------------------------------------
# 7. Softmax each row -> attention weights A. Every row sums to 1.
# ---------------------------------------------------------------------------
A = softmax(masked_scores)
print("\nA = softmax(masked scores) =\n", A)
print("row sums (must all be 1.0) =", A.sum(axis=1))

# ---------------------------------------------------------------------------
# 8. Output: weighted sum of the value vectors, O = A V, shape (n, d_v).
# ---------------------------------------------------------------------------
O = A @ V
print("\nO = A @ V =\n", O)

assert np.allclose(A.sum(axis=1), 1.0), "softmax rows must sum to 1"
print("\nCheck passed: every row of A sums to 1.")

assert O.shape == X.shape, "O must be the same shape as X to add back into the residual stream"
print(f"Check passed: O has shape {O.shape}, same as X {X.shape} — O can be added directly to X.")
