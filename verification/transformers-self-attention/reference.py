"""
Reference implementation: Scaled Dot-Product Attention and Multi-Head Attention.

Matches exactly the formulations in:
  Vaswani et al. (2017). "Attention Is All You Need." NeurIPS 2017.
  https://arxiv.org/abs/1706.03762

Equations reproduced:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V          [Eq. 1]
  MultiHead(Q,K,V)   = Concat(head_1,...,head_h) W^O         [Eq. 2]
  head_i             = Attention(Q W^Q_i, K W^K_i, V W^V_i)  [Eq. 3]

Toy example dimensions (kept small for hand-verifiability):
  seq_len = 3   tokens: ["The", "cat", "sat"]
  d_model = 8
  h       = 2   attention heads
  d_k     = d_v = d_model // h = 4  (per head)

All weight matrices are fixed via numpy seed 42 so outputs are deterministic.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fixed toy configuration
# ---------------------------------------------------------------------------
TOKENS   = ["The", "cat", "sat"]
SEQ_LEN  = 3
D_MODEL  = 8
H        = 2          # number of heads
D_K      = D_MODEL // H   # 4  — dimension of Q and K projections per head
D_V      = D_MODEL // H   # 4  — dimension of V projection per head
SEED     = 42

rng = np.random.default_rng(SEED)

# Input embeddings: (seq_len, d_model)
EMBEDDINGS = rng.standard_normal((SEQ_LEN, D_MODEL))

# Per-head weight matrices — shape (d_model, d_k) or (d_model, d_v)
W_Q = [rng.standard_normal((D_MODEL, D_K)) for _ in range(H)]
W_K = [rng.standard_normal((D_MODEL, D_K)) for _ in range(H)]
W_V = [rng.standard_normal((D_MODEL, D_V)) for _ in range(H)]

# Output projection: (h * d_v, d_model) = (8, 8)
W_O = rng.standard_normal((H * D_V, D_MODEL))


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax using the max-subtraction trick.

    For a vector x, softmax_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x)).
    Subtracting max(x) does not change the result (cancels in numerator and
    denominator) but prevents overflow when x_i are large.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> dict:
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    Args:
        Q: (seq_len, d_k)
        K: (seq_len, d_k)
        V: (seq_len, d_v)

    Returns dict with keys:
        scores   — raw dot products before scaling, shape (seq_len, seq_len)
        scaled   — scores / sqrt(d_k),             shape (seq_len, seq_len)
        weights  — softmax(scaled),                 shape (seq_len, seq_len)
        output   — weights @ V,                     shape (seq_len, d_v)
    """
    d_k = Q.shape[-1]
    scores  = Q @ K.T                         # (seq_len, seq_len)
    scaled  = scores / np.sqrt(d_k)           # divide by sqrt(d_k) for stability
    weights = softmax(scaled, axis=-1)         # row-wise softmax
    output  = weights @ V                      # (seq_len, d_v)
    return {"scores": scores, "scaled": scaled, "weights": weights, "output": output}


def causal_scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask_value: float = -1e9,
) -> dict:
    """
    Attention with a causal (look-ahead) mask.

    Used in decoder self-attention to prevent position i from attending to
    future positions j > i.  The upper triangle (k=1) of the score matrix
    is set to mask_value before softmax so exp(mask_value) ≈ 0.

    Args:
        Q, K, V:     same as scaled_dot_product_attention
        mask_value:  value written into upper triangle before softmax (-1e9 default)

    Returns dict with keys:
        scores  — raw dot products (unmasked)
        scaled  — scores / sqrt(d_k) (unmasked)
        masked  — scaled with upper triangle set to mask_value
        weights — softmax(masked), upper triangle ≈ 0
        output  — weights @ V
    """
    d_k = Q.shape[-1]
    n = Q.shape[0]
    scores  = Q @ K.T
    scaled  = scores / np.sqrt(d_k)

    # Causal mask: upper triangle (j > i) → -∞ approximation
    causal = np.triu(np.ones((n, n), dtype=bool), k=1)
    masked = scaled.copy()
    masked[causal] = mask_value

    weights = softmax(masked, axis=-1)
    output  = weights @ V
    return {"scores": scores, "scaled": scaled, "masked": masked,
            "weights": weights, "output": output}


def multi_head_attention(
    X: np.ndarray,
    W_Q: list,
    W_K: list,
    W_V: list,
    W_O: np.ndarray,
) -> dict:
    """
    Compute MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O.

    In self-attention Q = K = V = X (the same sequence attends to itself).

    Args:
        X:   (seq_len, d_model) — input (embeddings)
        W_Q: list of h matrices, each (d_model, d_k)
        W_K: list of h matrices, each (d_model, d_k)
        W_V: list of h matrices, each (d_model, d_v)
        W_O: (h * d_v, d_model)

    Returns dict with keys:
        heads   — list of per-head attention dicts (from scaled_dot_product_attention)
        concat  — concatenated head outputs, shape (seq_len, h * d_v)
        output  — concat @ W_O,              shape (seq_len, d_model)
    """
    h = len(W_Q)
    head_results = []
    for i in range(h):
        Q_i = X @ W_Q[i]   # (seq_len, d_k)
        K_i = X @ W_K[i]   # (seq_len, d_k)
        V_i = X @ W_V[i]   # (seq_len, d_v)
        head_results.append(scaled_dot_product_attention(Q_i, K_i, V_i))

    concat = np.concatenate([r["output"] for r in head_results], axis=-1)  # (seq_len, h*d_v)
    output = concat @ W_O                                                    # (seq_len, d_model)
    return {"heads": head_results, "concat": concat, "output": output}


# ---------------------------------------------------------------------------
# Run on toy example when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    result = multi_head_attention(EMBEDDINGS, W_Q, W_K, W_V, W_O)

    for i, head in enumerate(result["heads"]):
        print(f"\n--- Head {i} ---")
        print("Q:\n", EMBEDDINGS @ W_Q[i])
        print("K:\n", EMBEDDINGS @ W_K[i])
        print("V:\n", EMBEDDINGS @ W_V[i])
        print("Scaled scores:\n", head["scaled"])
        print("Attention weights:\n", head["weights"])
        print("Head output:\n", head["output"])

    print("\nConcatenated heads:\n", result["concat"])
    print("Final output:\n", result["output"])
