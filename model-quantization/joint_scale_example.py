"""
Verifies the "why not skip dequantizing between layers" slide: shows that
the raw INT32 matmul accumulator carries an arbitrary, layer-specific joint
scale (s_x * s_w), and that calibrating the NEXT layer's quantization scale
requires dividing by that joint scale first -- i.e. dequantization is what
produces the number the next layer's scale gets built from, not a step you
can skip just because requantization follows it.
"""
import numpy as np

X = np.array([[0.5, -1.2, 0.3], [2.0, 0.1, -0.7]])
W = np.array([[1.0, -0.5], [0.8, 2.0], [-1.5, 0.3]])


def quantize_symmetric(t, levels=127):
    s = levels / np.max(np.abs(t))
    q = np.round(s * t).astype(int)
    return q, s


Xq, sx = quantize_symmetric(X)
Wq, sw = quantize_symmetric(W)
print("X_q =\n", Xq, " s_x =", round(sx, 4))
print("W_q =\n", Wq, " s_w =", round(sw, 4))

Y_int = Xq @ Wq
print("\nY_int32 (raw accumulator) =\n", Y_int)

joint_scale = sx * sw
print(f"\njoint scale s_x*s_w = {joint_scale:.4f} -- specific to this layer, meaningless elsewhere")

Y_real = Y_int / joint_scale
Y_true = X @ W
print("\nY dequantized =\n", np.round(Y_real, 4))
print("Y true (no quantization) =\n", np.round(Y_true, 4))
print("max abs error:", np.max(np.abs(Y_real - Y_true)))

alpha_next = np.max(np.abs(Y_real))
alpha_raw = np.max(np.abs(Y_int))
print(f"\nnext layer's alpha = max|Y| = {alpha_next:.4f}  (only meaningful after dividing by the joint scale)")
print(f"max|Y_int32| alone = {alpha_raw}  -- not usable directly, old joint scale still baked in")
