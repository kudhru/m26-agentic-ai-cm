"""
Verifies the numeric claims transcribed onto the ReTool slides, against the
exact figures in the ReTool paper (Feng, Huang et al., ByteDance Seed,
arXiv:2504.11536), Table 1 and the Abstract/Section 3.2 prose.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("1. Headline comparison vs. the text-based RL baseline (Table 1)")
print("=" * 70)
retool_aime24, textrl_aime24 = 67.0, 40.0
retool_aime25, textrl_aime25 = 49.3, 36.7
retool_steps, textrl_steps = 400, 1080
print(f"AIME2024: ReTool {retool_aime24}% @ {retool_steps} steps  vs.  "
      f"Text-based RL {textrl_aime24}% @ {textrl_steps}+ steps")
print(f"  accuracy gap = +{retool_aime24 - textrl_aime24:.1f} points")
print(f"AIME2025: ReTool {retool_aime25}% vs. Text-based RL {textrl_aime25}%")
print(f"  accuracy gap = +{retool_aime25 - textrl_aime25:.1f} points")
print(f"ReTool reaches its (higher) score using only "
      f"{retool_steps/textrl_steps:.1%} of the text-based baseline's training steps")

print("\n" + "=" * 70)
print("2. Beating published baselines (paper's own deltas, Sec. 3.2)")
print("=" * 70)
s1_32b_aime24 = 56.7
o1_preview_aime25 = 37.9
print(f"AIME2024: ReTool {retool_aime24}% - s1-32B {s1_32b_aime24}% = "
      f"+{retool_aime24 - s1_32b_aime24:.1f} points (paper states '10.3%')")
print(f"AIME2025: ReTool {retool_aime25}% - o1-preview {o1_preview_aime25}% = "
      f"+{retool_aime25 - o1_preview_aime25:.1f} points (paper states '11.4%')")

o1_preview_aime24 = 44.6
retool_distill_aime24 = 72.5
print(f"AIME2024 (DeepSeek-R1-Distill-Qwen-32B backbone): "
      f"{retool_distill_aime24}% - o1-preview {o1_preview_aime24}% = "
      f"+{retool_distill_aime24 - o1_preview_aime24:.1f} points "
      f"(abstract states 'surpassing o1-preview by 27.9%')")

print("\n" + "=" * 70)
print("3. Ablation ladder on Qwen2.5-32B-Instruct (Table 1) -- where do the")
print("   gains actually come from?")
print("=" * 70)
base, cold_start, text_rl, retool_rl = 26.7, 40.9, 40.0, 67.0
print(f"base model (no training):            {base}%")
print(f"+ cold-start SFT only (no RL):        {cold_start}%  "
      f"(delta vs. base: +{cold_start - base:.1f})")
print(f"+ RL, text-only (no code interpreter): {text_rl}%  "
      f"(delta vs. base: +{text_rl - base:.1f})")
print(f"+ RL, with code interpreter (ReTool):  {retool_rl}%  "
      f"(delta vs. cold-start: +{retool_rl - cold_start:.1f}, "
      f"delta vs. text-RL: +{retool_rl - text_rl:.1f})")
print("-> cold-start SFT alone and text-only RL alone land within 1 point of")
print("   each other; the big jump only appears once RL is combined with tool use")

print("\n" + "=" * 70)
print("4. Response-length reduction after RL (Sec. 3.3, Figure 3a)")
print("=" * 70)
len_before, len_after = 10000, 6000
pct_shorter = (len_before - len_after) / len_before
print(f"~{len_before} tokens before RL -> ~{len_after} tokens after RL "
      f"= {pct_shorter:.0%} shorter (paper states '40% shorter')")

print("\n" + "=" * 70)
print("5. Code ratio and correct-code-count growth (Sec. 3.3, Figure 3b/d)")
print("=" * 70)
code_ratio_start, code_ratio_end = 18, 98
correct_code_start, correct_code_end = 1000, 5000
print(f"code ratio: ~{code_ratio_start}% of responses at step 40 -> "
      f"~{code_ratio_end}% by step 400")
print(f"correct code counts on test set: ~{correct_code_start} -> ~{correct_code_end} "
      f"({correct_code_end / correct_code_start:.0f}x)")
