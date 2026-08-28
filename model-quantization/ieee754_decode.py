"""
Step-by-step IEEE-754-style floating point decode, and derivation of each
format's representable min/max from its (sign, exponent, mantissa) bit
widths. Verifies the blog's own FP16 pi-encoding example, then generalizes
to FP32 and BF16, cross-checking every derived min/max against numpy's own
finfo() ground truth.
"""
import struct
import numpy as np

np.set_printoptions(precision=10)


def decode(sign_bit, exp_bits, mantissa_bits, bias, label):
    """General (-1)^S * 2^(E-bias) * (1 + F/2^m) decoder, printed step by step."""
    exp_val = int(exp_bits, 2)
    mant_val = int(mantissa_bits, 2)
    m = len(mantissa_bits)
    fraction = mant_val / (2 ** m)
    unbiased_exp = exp_val - bias
    sign = (-1) ** sign_bit
    value = sign * (2 ** unbiased_exp) * (1 + fraction)
    print(f"  {label}")
    print(f"    sign bit = {sign_bit}  ->  (-1)^{sign_bit} = {sign}")
    print(f"    exponent bits = {exp_bits} = {exp_val} (decimal),  bias = {bias}")
    print(f"    unbiased exponent = {exp_val} - {bias} = {unbiased_exp}")
    print(f"    mantissa bits = {mantissa_bits} -> sum of 2^-k for each set bit = {mant_val}/2^{m} = {fraction}")
    print(f"    value = {sign} * 2^{unbiased_exp} * (1 + {fraction}) = {value}")
    return value


print("=" * 70)
print("1. Reproduce the blog's own FP16 encoding of pi")
print("=" * 70)
v = decode(0, "10000", "1001001000", bias=15, label="FP16: 0 10000 1001001000")
print(f"  -> decoded value = {v}")
assert abs(v - 3.140625) < 1e-9
print("  matches blog's stated 3.140625. Verified.\n")

print("1b. Decode FP16's own max bit pattern (all mantissa bits set, largest usable exponent)")
v_max = decode(0, "11110", "1111111111", bias=15, label="FP16 max: 0 11110 1111111111")
print(f"  -> decoded value = {v_max}")
assert abs(v_max - 65504.0) < 1e-9
print("  matches FP16's published max, 65504. Verified.\n")

print("cross-check via struct (pack pi as float16, read back raw bits):")
raw = np.float16(np.pi)
bits = np.frombuffer(np.array([raw]).tobytes(), dtype=np.uint16)[0]
s = (bits >> 15) & 0x1
e = (bits >> 10) & 0x1F
f = bits & 0x3FF
print(f"  np.float16(pi) = {float(raw)}, raw bits = {bits:016b} -> "
      f"sign={s} exp={e:05b} frac={f:010b}")
assert f"{s}" == "0" and f"{e:05b}" == "10000" and f"{f:010b}" == "1001001000"
print("  matches the hand-decoded bit pattern exactly. Verified.\n")

print("=" * 70)
print("2. Derive min/max representable magnitude from bit-width allocation")
print("=" * 70)


def format_range(exp_bits, mantissa_bits, name):
    bias = 2 ** (exp_bits - 1) - 1
    # normal numbers: stored exponent from 1 to (2^exp_bits - 2); all-0s and
    # all-1s exponents are reserved (subnormals / inf & nan)
    e_max = 2 ** exp_bits - 2
    e_min = 1
    max_val = (2 - 2 ** (-mantissa_bits)) * 2 ** (e_max - bias)
    min_normal = 2 ** (e_min - bias)
    print(f"  {name}: sign=1, exponent={exp_bits} bits, mantissa={mantissa_bits} bits")
    print(f"    bias = 2^({exp_bits}-1) - 1 = {bias}")
    print(f"    largest usable stored exponent = 2^{exp_bits}-2 = {e_max}  "
          f"(all-1s reserved for inf/nan) -> unbiased = {e_max - bias}")
    print(f"    max = (2 - 2^-{mantissa_bits}) * 2^{e_max - bias} = {max_val:.6e}")
    print(f"    smallest usable stored exponent = 1 -> unbiased = {e_min - bias}")
    print(f"    min (normal) = 1.0 * 2^{e_min - bias} = {min_normal:.6e}")
    return max_val, min_normal


fp32_max, fp32_min = format_range(8, 23, "FP32")
fp16_max, fp16_min = format_range(5, 10, "FP16")
bf16_max, bf16_min = format_range(8, 7, "BF16")

print("\ncross-check against numpy's own finfo() ground truth:")
for name, dtype, derived_max, derived_min in [
    ("FP32", np.float32, fp32_max, fp32_min),
    ("FP16", np.float16, fp16_max, fp16_min),
]:
    info = np.finfo(dtype)
    print(f"  {name}: derived max={derived_max:.6e} vs finfo.max={info.max:.6e}  "
          f"derived min_normal={derived_min:.6e} vs finfo.tiny={info.tiny:.6e}")
    assert abs(derived_max - info.max) / info.max < 1e-6
    assert abs(derived_min - info.tiny) / info.tiny < 1e-6
print("  FP32 and FP16 derived ranges match numpy finfo exactly. Verified.")
print(f"  BF16 derived max={bf16_max:.6e} (no native numpy dtype to cross-check;"
      f" matches known bfloat16 max 3.3895e+38).")

print("\n" + "=" * 70)
print("3. Reproduce the blog's overflow/underflow table (75505.0 and 1.8e-42)")
print("=" * 70)
for val in [75505.0, 1.8e-42]:
    v64 = np.float64(val)
    v32 = np.float32(val)
    v16 = np.float16(val)
    print(f"  original={val}  ->  64-bit={v64}  32-bit={v32}  16-bit={v16}")
