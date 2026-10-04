"""
Verifies the numeric claims transcribed onto the DeepSeekMath slides, against
the exact figures in the DeepSeekMath paper (Shao et al., arXiv:2402.03300).
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("1. DeepSeekMath Corpus vs. OpenWebMath, size ratio (Table 1)")
print("=" * 70)
deepseekmath_corpus_B = 120.2
openwebmath_B = 13.6
ratio = deepseekmath_corpus_B / openwebmath_B
print(f"{deepseekmath_corpus_B}B / {openwebmath_B}B = {ratio:.2f}x  (paper states 'almost 9x')")

print("\n" + "=" * 70)
print("2. GRPO improvement: DeepSeekMath-Instruct -> DeepSeekMath-RL (Table 5)")
print("=" * 70)
pairs = {
    "GSM8K (CoT)": (82.9, 88.2),
    "MATH (CoT)": (46.8, 51.7),
    "CMATH (CoT)": (84.6, 88.8),
}
for name, (before, after) in pairs.items():
    print(f"{name}: {before} -> {after}  (delta = +{after - before:.1f} points)")
