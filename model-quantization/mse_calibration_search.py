"""
Verifies the MSE-based calibration search used on the "Calibrating by
search: minimize the MSE" slide. Simulates a heavier-tailed (Laplace)
weight-like tensor, sweeps candidate clip thresholds, quantizes+dequantizes
under each, and finds the threshold that minimizes reconstruction MSE
against the original (unclipped) data -- then compares the improvement at
INT4 vs INT8 to show the search matters more at aggressive bit-widths.
"""
import numpy as np

rng = np.random.default_rng(3)
x = rng.laplace(0, 1.0, size=5000)
true_max = np.max(np.abs(x))
n = len(x)
print(f"n={n} samples, true max |x| = {true_max:.4f}")


def quantize_symmetric(x, clip, levels):
    xc = np.clip(x, -clip, clip)
    s = levels / clip
    q = np.round(s * xc)
    return q / s


def mse_search(x, true_max, levels, fracs):
    mses = []
    for f in fracs:
        c = f * true_max
        xhat = quantize_symmetric(x, c, levels)
        mses.append(np.mean((x - xhat) ** 2))
    best_i = int(np.argmin(mses))
    return fracs, mses, best_i


fracs = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.85, 1.0])

print("\n--- INT4 symmetric (levels=7) ---")
fr, mses, best_i = mse_search(x, true_max, 7, fracs)
for f, m in zip(fr, mses):
    tag = "  <-- best" if f == fr[best_i] else ""
    print(f"  clip={f*100:5.1f}% of max ({f*true_max:.4f})  MSE={m:.5f}{tag}")
reduction = 100 * (1 - mses[best_i] / mses[-1])
pct = 100 * np.mean(np.abs(x) <= fr[best_i] * true_max)
print(f"best clip corresponds to the {pct:.1f}th percentile of |x|")
print(f"MSE reduction vs naive full range: {reduction:.1f}%")

print("\n--- INT8 symmetric (levels=127), for comparison ---")
fr8, mses8, best_i8 = mse_search(x, true_max, 127, fracs)
for f, m in zip(fr8, mses8):
    tag = "  <-- best" if f == fr8[best_i8] else ""
    print(f"  clip={f*100:5.1f}% of max ({f*true_max:.4f})  MSE={m:.5f}{tag}")
print(f"INT8 best clip = {fr8[best_i8]*100:.0f}% of max (search barely matters at this resolution)")
