"""
Worked example for the general affine quantize/dequantize formula on the
"What is quantization?" slide: x_int = clamp(round(x/s)+z, 0, 2^b-1),
x_hat = s*(x_int - z). Uses a calibrated range [min,max] = [-2.0, 6.0] with
b=4 bits (16 levels), matching this slide's earlier "4 bits -> 16 levels"
statement, and includes an out-of-range value to demonstrate why clamp()
is needed.
"""
import math

lo, hi, b = -2.0, 6.0, 4
n_levels = 2 ** b
s = (hi - lo) / (n_levels - 1)
z = round(-lo / s)
print(f"range=[{lo},{hi}]  b={b}  levels={n_levels}")
print(f"s = (hi-lo)/(2^b-1) = {hi-lo}/{n_levels-1} = {s:.4f}")
print(f"z = round(-lo/s) = round({-lo}/{s:.4f}) = round({-lo/s:.4f}) = {z}")


def quantize(x):
    raw = round(x / s) + z
    clamped = max(0, min(n_levels - 1, raw))
    return raw, clamped


def dequantize(x_int):
    return s * (x_int - z)


for x in [-2.0, 0.0, 3.2, 6.0, 9.0]:
    raw, x_int = quantize(x)
    x_hat = dequantize(x_int)
    clipped = " (clamped!)" if raw != x_int else ""
    print(f"x={x:+.2f}  x/s={x/s:+.4f}  round+z={raw}{clipped}  x_int={x_int}  "
          f"x_hat=s*(x_int-z)={s:.4f}*({x_int}-{z})={x_hat:+.4f}  err={abs(x-x_hat):.4f}")
