"""
Verifies the GPTQ / Optimal-Brain-Surgeon compensation formula used across
the GPTQ slides:
  - H = X @ X.T from calibration activations (not from the weights)
  - the exact per-step update delta = (e / [Hinv]_qq) * Hinv[:, q],
    applied as w <- w - delta, which forces the just-quantized coordinate
    exactly onto its quantized value and optimally compensates the rest
  - the row-by-row trace used on the "one weight at a time" slide
  - the 64-weight aggregate result used on the "does this actually help" slide
"""
import numpy as np

# ---------------------------------------------------------------------------
# Part 1: small 3-weight row -- exact step-by-step trace shown on the slides
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
d_in, n = 3, 200
X = rng.normal(0, 1, size=(d_in, n))
X[1] = 2.5 * X[0] + 0.3 * rng.normal(0, 1, size=n)

H = X @ X.T
H += 0.01 * np.mean(np.diag(H)) * np.eye(d_in)
print("H =\n", np.round(H, 2))
print("H^-1 =\n", np.round(np.linalg.inv(H), 4))

w = np.array([0.42, 0.30, 0.50])
LEVELS = 7


def quantize(vals, scale):
    return np.round(scale * vals) / scale


scale = LEVELS / np.max(np.abs(w))
w_naive = quantize(w, scale)

w_gptq = w.copy()
remaining = [0, 1, 2]
print(f"\nstart: {np.round(w_gptq, 4)}")
for q_idx in [0, 1, 2]:
    val = w_gptq[q_idx]
    q_val = quantize(np.array([val]), scale)[0]
    e = val - q_val
    H_work = H[np.ix_(remaining, remaining)]
    Hinv_r = np.linalg.inv(H_work)
    pos = remaining.index(q_idx)
    delta = (e / Hinv_r[pos, pos]) * Hinv_r[:, pos]
    for k, idx in enumerate(remaining):
        w_gptq[idx] -= delta[k]
    w_gptq[q_idx] = q_val
    print(f"step {q_idx+1}: w={val:.4f} -> q={q_val:.4f}  e={e:.4f}  row now {np.round(w_gptq, 4)}")
    remaining.remove(q_idx)

print(f"\nfinal naive: {np.round(w_naive, 4)}")
print(f"final gptq : {np.round(w_gptq, 4)}  (matches naive at this small scale)")

# ---------------------------------------------------------------------------
# Part 2: 64-weight row -- the aggregate payoff shown on "does it actually help"
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
d_in, n = 64, 512
X = rng.normal(0, 1, size=(d_in, n))
mix = rng.normal(0, 0.3, size=(d_in, d_in))
X = X + mix @ X
H = X @ X.T
H += 0.01 * np.mean(np.diag(H)) * np.eye(d_in)

w = rng.normal(0, 1, size=d_in) * 0.3
scale = LEVELS / np.max(np.abs(w))

w_naive = quantize(w, scale)
err_naive = w - w_naive
loss_naive = err_naive @ H @ err_naive

w_gptq = w.copy()
remaining = list(range(d_in))
for q_idx in range(d_in):
    val = w_gptq[q_idx]
    q_val = quantize(np.array([val]), scale)[0]
    e = val - q_val
    H_work = H[np.ix_(remaining, remaining)]
    Hinv_r = np.linalg.inv(H_work)
    pos = remaining.index(q_idx)
    delta = (e / Hinv_r[pos, pos]) * Hinv_r[:, pos]
    for k, idx in enumerate(remaining):
        w_gptq[idx] -= delta[k]
    w_gptq[q_idx] = q_val
    remaining.remove(q_idx)

err_gptq = w - w_gptq
loss_gptq = err_gptq @ H @ err_gptq

print(f"\n--- 64-weight row ---")
print(f"naive loss (e^T H e) = {loss_naive:.4f}")
print(f"gptq  loss (e^T H e) = {loss_gptq:.4f}  ({100*(1-loss_gptq/loss_naive):.1f}% lower)")
print(f"plain MSE -- naive: {np.mean(err_naive**2):.6f}  gptq: {np.mean(err_gptq**2):.6f}")
