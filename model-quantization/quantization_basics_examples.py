"""
Verifies the symmetric- and asymmetric-quantization worked examples used in
Part 2 of the deck, reproducing the article's own example vector exactly:
x = [5.47, 3.08, -7.59, 0, -1.95, -4.57, 10.8]
"""
import numpy as np

np.set_printoptions(suppress=True)

x = np.array([5.47, 3.08, -7.59, 0, -1.95, -4.57, 10.8])
print("x =", x.tolist())

print("\n--- Symmetric (absmax) quantization, INT8 range [-127,127] ---")
alpha = np.max(np.abs(x))
s = 127 / alpha
print(f"alpha = max(|x|) = {alpha}")
print(f"s = 127/alpha = 127/{alpha} = {s:.4f}")
x_q = np.round(s * x).astype(int)
print("x_quantized = round(s*x) =", x_q.tolist())
x_hat = x_q / s
print("x_dequantized = x_quantized/s =", np.round(x_hat, 4).tolist())
err = x - x_hat
print("quantization error =", np.round(err, 4).tolist())

print("\n--- Asymmetric (zero-point) quantization, INT8 range [-128,127] ---")
beta = np.min(x)
alpha2 = np.max(x)
s2 = 255 / (alpha2 - beta)
z = round(-s2 * beta) - 128
print(f"beta = min(x) = {beta}, alpha = max(x) = {alpha2}")
print(f"s = 255/(alpha-beta) = 255/{alpha2 - beta} = {s2:.4f}")
print(f"z = round(-s*beta) - 128 = round({-s2*beta:.4f}) - 128 = {z}")
x_q2 = np.clip(np.round(s2 * x + z), -128, 127).astype(int)
print("x_quantized = clip(round(s*x+z), -128, 127) =", x_q2.tolist())
x_hat2 = (x_q2 - z) / s2
print("x_dequantized = (x_quantized - z)/s =", np.round(x_hat2, 4).tolist())

print("\n--- Outlier vector: symmetric quantization without clipping ---")
xo = np.array([-.59, -.21, -.07, .13, .28, .57, 256])
alpha_o = np.max(np.abs(xo))
s_o = 127 / alpha_o
xo_q = np.round(s_o * xo).astype(int)
print("x =", xo.tolist())
print(f"alpha = {alpha_o}, s = 127/{alpha_o} = {s_o:.6f}")
print("x_quantized (no clipping) =", xo_q.tolist())

print("\n--- Same outlier vector, WITH clipping to [-5,5] before quantizing ---")
clip_lo, clip_hi = -5, 5
xo_clipped = np.clip(xo, clip_lo, clip_hi)
alpha_c = clip_hi
s_c = 127 / alpha_c
xo_c_q = np.round(s_c * xo_clipped).astype(int)
print(f"clip range = [{clip_lo},{clip_hi}]")
print("x clipped =", xo_clipped.tolist())
print(f"alpha = {alpha_c}, s = 127/{alpha_c} = {s_c:.4f}")
print("x_quantized (with clipping) =", xo_c_q.tolist())
