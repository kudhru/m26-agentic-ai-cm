"""
Numeric worked examples for peft-notes.html, verified with numpy before being
transcribed into the notes. Covers: LoRA forward pass + merge, DoRA's
magnitude/direction decomposition, weight normalization, and SVD / the
Eckart-Young-Mirsky best-low-rank-approximation theorem (background for
AdaLoRA), plus a small NF4-vs-uniform quantization comparison.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. LoRA: forward pass, merge, and a non-degenerate example input
# ---------------------------------------------------------------------------
hr("1. LoRA worked example")

W0 = np.array([
    [2., 1.],
    [0., 1.],
    [1., 2.],
])  # d=3, d_out=2, frozen pretrained weight
B = np.array([[1.], [1.], [0.]])  # d x r, r=1
A = np.array([[1., 2.]])          # r x d_out
alpha = 2.0
r = 1
scale = alpha / r

delta_W = B @ A
W_merged = W0 + scale * delta_W

x = np.array([1., 1., 1.])  # one input row

h_direct = x @ W0 + scale * (x @ B @ A)
h_merged = x @ W_merged

print("W0 =\n", W0)
print("B =\n", B)
print("A =\n", A)
print("alpha/r =", scale)
print("delta_W = B @ A =\n", delta_W)
print("W_merged = W0 + (alpha/r) * delta_W =\n", W_merged)
print("x =", x)
print("x @ W0 =", x @ W0)
print("x @ B =", x @ B)
print("(x @ B) @ A =", (x @ B) @ A)
print("h (via separate B,A) =", h_direct)
print("h (via merged W') =", h_merged)
assert np.allclose(h_direct, h_merged)
print("Check passed: separate and merged computations agree.")

# ---------------------------------------------------------------------------
# 2. DoRA: magnitude/direction decomposition, using the same W0 and the same
#    LoRA-updated direction V' = W0 + (alpha/r) B A from above.
# ---------------------------------------------------------------------------
hr("2. DoRA worked example")


def col_norms(M):
    return np.linalg.norm(M, axis=0)


m0 = col_norms(W0)
V0_normalized = W0 / m0
reconstructed = m0 * V0_normalized
print("Column norms of W0, ||W0||_c =", m0)
print("W0 / ||W0||_c =\n", V0_normalized)
print("m0 * (W0/||W0||_c) =\n", reconstructed)
assert np.allclose(reconstructed, W0)
print("Check passed: magnitude x direction reconstructs W0 exactly.")

V_prime = W_merged  # V' = W0 + (alpha/r) B A, computed in section 1
m_passive = col_norms(V_prime)
print("\nV' = W0 + (alpha/r) B A =\n", V_prime)
print("Column norms of V', ||V'||_c =", m_passive)

# If m were left passive (just the column norm of V'), DoRA's weight equals
# plain LoRA's merged weight exactly:
W_dora_passive = m_passive * (V_prime / m_passive)
assert np.allclose(W_dora_passive, V_prime)
print("If m is left passive (= ||V'||_c), W_DoRA == W' (plain LoRA). Verified.")

# Now show DoRA's actual freedom: train m independently, different from ||V'||_c
m_learned = np.array([5.0, 6.5])  # illustrative "trained" magnitude, != m_passive
W_dora = m_learned * (V_prime / m_passive)
print("\nSuppose training found an independent magnitude m_learned =", m_learned)
print("(compare to the passive value ||V'||_c =", m_passive, ")")
print("W_DoRA = m_learned * (V'/||V'||_c) =\n", W_dora)
print("vs. plain LoRA's W' =\n", V_prime)
print("These differ -- DoRA reaches weight matrices plain LoRA cannot, from the same B,A.")

# ---------------------------------------------------------------------------
# 3. Weight normalization: the trivial magnitude/direction identity
# ---------------------------------------------------------------------------
hr("3. Weight normalization identity")

v = np.array([3., 4.])
norm_v = np.linalg.norm(v)
direction = v / norm_v
print("v =", v)
print("||v|| =", norm_v)
print("v / ||v|| =", direction)
print("||v|| * (v/||v||) =", norm_v * direction)
assert np.allclose(norm_v * direction, v)
print("Check passed.")

# ---------------------------------------------------------------------------
# 4. SVD and the Eckart-Young-Mirsky theorem (background for AdaLoRA)
# ---------------------------------------------------------------------------
hr("4. SVD / best rank-1 approximation")

M = np.array([
    [3., 1., 2.],
    [2., 4., 1.],
    [1., 1., 5.],
])
U, S, Vt = np.linalg.svd(M)
print("M =\n", M)
print("Singular values sigma =", S)

r = 1
M_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
err_svd = np.linalg.norm(M - M_r, ord="fro")
print(f"\nBest rank-{r} approx via truncated SVD, M_{r} =\n", M_r)
print(f"Frobenius error ||M - M_{r}||_F =", err_svd)

# Compare against a different, arbitrary rank-1 matrix (same rank, not from SVD) --
# e.g. "just keep the top-left entry and zero everything else"
M_arb = np.zeros_like(M)
M_arb[0, 0] = M[0, 0]
err_arb = np.linalg.norm(M - M_arb, ord="fro")
print(f"\nAn arbitrary alternative rank-1 matrix, M_arb =\n", M_arb)
print("Frobenius error ||M - M_arb||_F =", err_arb)
print(f"\nSVD error ({err_svd:.4f}) < arbitrary rank-1 error ({err_arb:.4f}):", err_svd < err_arb)

# ---------------------------------------------------------------------------
# 5. The actual NF4 table (Dettmers et al. 2023, QLoRA, Sec 3 + Eq 4), matching
#    the paper's asymmetric construction and the bitsandbytes reference
#    implementation -- NOT a simple symmetric quantile-per-level formula. A
#    symmetric k-bit quantile grid has no exact zero (Eq 4, applied directly,
#    would never land on 0), so NF4 builds two half-tables at reduced
#    resolution -- 2^(k-1) quantiles for the negative side, 2^(k-1)+1 for the
#    positive side -- and merges them, dropping the shared zero.
# ---------------------------------------------------------------------------
hr("5. The real NF4 table (asymmetric construction)")

from statistics import NormalDist

nd = NormalDist()


def linspace_list(start, stop, n):
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def create_nf4_table(offset=0.9677083, k=4):
    half_pos = 2 ** (k - 1) + 1  # 9 for k=4
    half_neg = 2 ** (k - 1)      # 8 for k=4
    probs_pos = linspace_list(offset, 0.5, half_pos)[:-1]  # 8 probabilities
    v1 = [nd.inv_cdf(p) for p in probs_pos]                # 8 positive values
    probs_neg = linspace_list(offset, 0.5, half_neg)[:-1]  # 7 probabilities
    v3 = [-nd.inv_cdf(p) for p in probs_neg]               # 7 negative values
    v = sorted(v1 + [0.0] + v3)                            # 16 values, exact 0 included
    m = max(abs(x) for x in v)
    return np.array([x / m for x in v])


NF4 = create_nf4_table()
print("NF4 table (index -> value):")
for i, x in enumerate(NF4):
    print(f"  {format(i, '04b')}  {x: .6f}")

uniform_levels = np.linspace(-1, 1, 16)
print("\nUniform 4-bit levels for comparison:\n", np.round(uniform_levels, 4))


def quantize(x, levels):
    idx = np.argmin(np.abs(x[:, None] - levels[None, :]), axis=1)
    return levels[idx]


sample_weights = np.array([0.05, -0.12, 0.31, -0.55, 0.02, 0.88, -0.03, 0.15])
print("\nSample weights (already rescaled into [-1,1]):", sample_weights)

nf4_hat = quantize(sample_weights, NF4)
uniform_hat = quantize(sample_weights, uniform_levels)
nf4_err = np.abs(sample_weights - nf4_hat)
uniform_err = np.abs(sample_weights - uniform_hat)
print("\nNF4 dequantized:      ", np.round(nf4_hat, 4))
print("NF4 abs error:        ", np.round(nf4_err, 4), " mean =", round(nf4_err.mean(), 4))
print("Uniform dequantized:  ", np.round(uniform_hat, 4))
print("Uniform abs error:    ", np.round(uniform_err, 4), " mean =", round(uniform_err.mean(), 4))
print(f"\nMean error on this one 8-weight sample -- NF4: {nf4_err.mean():.4f} vs "
      f"uniform: {uniform_err.mean():.4f} (too small a sample to show NF4's real "
      f"advantage reliably -- see the large-sample check in section 6).")

# ---------------------------------------------------------------------------
# 6. NF4 vs. uniform, averaged over many realistic blocks (the actual claim:
#    NF4 is optimal in *expectation* over normally-distributed weights, not
#    guaranteed to win on any single hand-picked block).
# ---------------------------------------------------------------------------
hr("6. NF4 vs. uniform -- large-sample mean absolute error")

rng = np.random.default_rng(0)
n_blocks = 20000
block_size = 64
nf4_errs, unif_errs = [], []
for _ in range(n_blocks):
    w = rng.standard_normal(block_size)
    s = np.max(np.abs(w))
    t = w / s
    nf4_hat_b = quantize(t, NF4) * s
    unif_hat_b = quantize(t, uniform_levels) * s
    nf4_errs.append(np.mean(np.abs(w - nf4_hat_b)))
    unif_errs.append(np.mean(np.abs(w - unif_hat_b)))

nf4_mean = np.mean(nf4_errs)
unif_mean = np.mean(unif_errs)
print(f"{n_blocks} blocks of {block_size} weights ~ N(0,1) each, absmax-rescaled per block:")
print(f"  mean abs error, NF4:     {nf4_mean:.4f}")
print(f"  mean abs error, uniform: {unif_mean:.4f}")
print(f"  NF4 reduces mean abs error by {100*(1 - nf4_mean/unif_mean):.1f}% vs. uniform")
