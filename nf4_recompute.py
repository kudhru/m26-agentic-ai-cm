"""
Recomputes every NF4 numeric example in peft-slides.html / peft-notes.html
using the QLoRA paper's actual asymmetric NF4 construction (Dettmers et al.
2023, arXiv:2305.14314, Sec 3, "4-bit NormalFloat Quantization" + Eq 4),
matching the bitsandbytes create_normal_map() reference implementation.
Replaces the earlier, incorrect symmetric-single-quantile approximation.
"""
from statistics import NormalDist

nd = NormalDist()


def linspace(start, stop, n):
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def create_normal_map(offset=0.9677083, k=4):
    half_pos = 2 ** (k - 1) + 1
    half_neg = 2 ** (k - 1)
    probs_pos = linspace(offset, 0.5, half_pos)[:-1]
    v1 = [nd.inv_cdf(p) for p in probs_pos]
    probs_neg = linspace(offset, 0.5, half_neg)[:-1]
    v3 = [-nd.inv_cdf(p) for p in probs_neg]
    v = sorted(v1 + [0.0] + v3)
    m = max(abs(x) for x in v)
    return [x / m for x in v]


NF4 = create_normal_map()
print("NF4 table (index -> value):")
for i, x in enumerate(NF4):
    print(f"  {format(i,'04b')}  {x: .6f}")


def nearest(x, levels):
    return min(range(len(levels)), key=lambda i: abs(levels[i] - x))


def uniform_levels(k=4):
    n = 2 ** k
    return linspace(-1, 1, n)


UNIF = uniform_levels()

print("\n--- Slide: 'Getting a weight back' single example ---")
idx = 8  # 1000
level = NF4[idx]
s = 0.88
w_hat = level * s
print(f"index=1000 (8) -> level={level:.6f} -> x s={s} -> what={level:.6f}*{s}={w_hat:.6f}")

print("\n--- Slide: 'QLoRA: a worked example' (8-weight block) ---")
w = [0.05, -0.12, 0.31, -0.55, 0.02, 0.88, -0.03, 0.15]
s = max(abs(x) for x in w)
print("s =", s)
errs = []
for wi in w:
    t = wi / s
    idx = nearest(t, NF4)
    level = NF4[idx]
    what = level * s
    err = abs(wi - what)
    errs.append(err)
    print(f"  w={wi:+.2f}  w/s={t:+.3f}  idx={format(idx,'04b')}  level={level:+.4f}  what={what:+.4f}  err={err:.4f}")
mae = sum(errs) / len(errs)
print("mean abs error =", round(mae, 4))
bits_lo = 8 * 4 + 32
bits_fp16 = 8 * 16
print(f"storage: {bits_lo} bits (4-bit x8 + 32-bit scale) vs {bits_fp16} bits fp16")

print("\n--- Slide: 'QLoRA: NF4 in action' (NF4 vs uniform, 5 weights) ---")
w5 = [0.05, -0.12, 0.31, -0.55, 0.88]
nf4_errs, unif_errs = [], []
for wi in w5:
    inf = nearest(wi, NF4)
    iuf = nearest(wi, UNIF)
    nf4_hat = NF4[inf]
    unif_hat = UNIF[iuf]
    e_nf4 = abs(wi - nf4_hat)
    e_unif = abs(wi - unif_hat)
    nf4_errs.append(e_nf4)
    unif_errs.append(e_unif)
    print(f"  x={wi:+.2f}  NF4->{nf4_hat:+.4f} (err {e_nf4:.4f})   uniform->{unif_hat:+.4f} (err {e_unif:.4f})")
print("mean abs error: NF4 =", round(sum(nf4_errs)/len(nf4_errs), 4), " uniform =", round(sum(unif_errs)/len(unif_errs), 4))

print("\n--- Notes §quantization: NF4 vs uniform (8-weight sample, from peft_worked_examples.py) ---")
sample_weights = [0.05, -0.12, 0.31, -0.55, 0.02, 0.88, -0.03, 0.15]
nf4_errs2, unif_errs2 = [], []
nf4_hats, unif_hats = [], []
for wi in sample_weights:
    inf = nearest(wi, NF4)
    iuf = nearest(wi, UNIF)
    nf4_hat = NF4[inf]
    unif_hat = UNIF[iuf]
    nf4_hats.append(nf4_hat)
    unif_hats.append(unif_hat)
    nf4_errs2.append(abs(wi - nf4_hat))
    unif_errs2.append(abs(wi - unif_hat))
print("NF4 dequantized:", [round(x,4) for x in nf4_hats])
print("NF4 abs err:", [round(x,4) for x in nf4_errs2], "mean =", round(sum(nf4_errs2)/len(nf4_errs2),4))
print("Uniform dequantized:", [round(x,4) for x in unif_hats])
print("Uniform abs err:", [round(x,4) for x in unif_errs2], "mean =", round(sum(unif_errs2)/len(unif_errs2),4))
