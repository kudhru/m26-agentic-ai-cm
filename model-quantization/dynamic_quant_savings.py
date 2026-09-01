"""
Verifies the memory/bandwidth numbers on the "If we dequantize right back,
what's saved?" slide: for one 4096x4096 projection applied to a 128-token
batch, how many bytes actually move into the matmul under INT8 quantization
vs staying FP32 throughout, and that the dequantized output Y is the same
size either way (it's a fresh, reduced tensor -- not a reconstruction of X).
"""
n, d_in, d_out = 128, 4096, 4096

x_fp32 = n * d_in * 4
x_int8 = n * d_in * 1
w_fp32 = d_in * d_out * 4
w_int8 = d_in * d_out * 1
y_fp32 = n * d_out * 4

print(f"X: {n}x{d_in}  FP32={x_fp32/1e6:.2f} MB  INT8={x_int8/1e6:.2f} MB  ({x_fp32/x_int8:.0f}x smaller)")
print(f"W: {d_in}x{d_out}  FP32={w_fp32/1e6:.2f} MB  INT8={w_int8/1e6:.2f} MB  ({w_fp32/w_int8:.0f}x smaller)")
print(f"Y: {n}x{d_out}  FP32={y_fp32/1e6:.2f} MB  (same size as X's FP32 form -- a fresh tensor, not a reconstruction of X)")

moved_int8 = x_int8 + w_int8
moved_fp32 = x_fp32 + w_fp32
print(f"\nbytes read into the matmul: INT8 path = {moved_int8:,} ({moved_int8/1e6:.2f} MB)")
print(f"                             FP32 path = {moved_fp32:,} ({moved_fp32/1e6:.2f} MB)")
print(f"reduction: {100*(1 - moved_int8/moved_fp32):.0f}%")
