"""
Verifies the numeric worked examples used on the DeepSeek-R1 slides, against
the exact reward formulas in the DeepSeek-R1 paper (DeepSeek-AI,
arXiv:2501.12948): Reward_rule = Reward_acc + Reward_format (Eq. 4), the GRPO
advantage formula (Eq. 3, identical in form to DeepSeekMath's), and the
language-consistency reward (Eq. 7).
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("1. Combined rule reward (accuracy + format) -> GRPO advantage")
print("   Reward_rule = Reward_acc + Reward_format  (paper Eq. 4)")
print("=" * 70)
acc = np.array([1, 1, 0, 0])
fmt = np.array([1, 0, 1, 0])
reward = acc + fmt
mean = reward.mean()
std = reward.std(ddof=1)
adv = (reward - mean) / std
print(f"rollout:            1     2     3     4")
print(f"accuracy reward: {acc.tolist()}")
print(f"format reward:   {fmt.tolist()}")
print(f"combined reward = acc + format: {reward.tolist()}")
print(f"mean={mean}, sample std (ddof=1)={std:.4f}")
print(f"GRPO advantage (Eq. 3): {np.round(adv, 4).tolist()}")
print("(rollouts 2 and 3, tied at reward=1 = the group average, get advantage")
print(" exactly 0 -- no gradient signal from them, despite very different")
print(" behavior: one got the answer right but skipped the <think> tags, the")
print(" other used the tags correctly but got the answer wrong)")

print("\n" + "=" * 70)
print("2. Language consistency reward (Eq. 7)")
print("   Reward_language = Num(Words_target) / Num(Words)")
print("=" * 70)
total_words_a, target_words_a = 50, 45
reward_lang_a = target_words_a / total_words_a
print(f"mild mixing: {target_words_a} of {total_words_a} words in the target language")
print(f"  Reward_language = {target_words_a}/{total_words_a} = {reward_lang_a:.4f}")

total_words_b, target_words_b = 60, 33
reward_lang_b = target_words_b / total_words_b
print(f"heavy mixing: {target_words_b} of {total_words_b} words in the target language")
print(f"  Reward_language = {target_words_b}/{total_words_b} = {reward_lang_b:.4f}")

print("\n" + "=" * 70)
print("3. Pass@1 vs. Pass@64 gap on AIME 2024 (paper's own headline numbers)")
print("=" * 70)
pass1, pass64 = 79.8, 90.0
print(f"Pass@1 = {pass1}%, Pass@64 = {pass64}%, gap = {pass64 - pass1:.1f} points")
print("(independently sampling more reasoning chains finds correct solutions")
print(" DeepSeek-R1 itself doesn't always reach in a single attempt)")
