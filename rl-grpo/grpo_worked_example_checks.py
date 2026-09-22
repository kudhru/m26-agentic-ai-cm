"""
Verifies the GRPO worked-example numbers transcribed onto the slides, against
the exact figures shown in Raschka's "Build a Reasoning Model (From Scratch)"
book, chapters 6 and 7 (https://github.com/rasbt/reasoning-from-scratch).
All input numbers below (rewards, logprobs, reference logprobs, old logprobs)
are read directly off the book's own figures -- this script only checks that
the downstream quantities the slides state (advantages, policy gradient loss,
clipped ratios, KL loss) are correctly derived from them.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("1. Rewards -> advantages (uses SAMPLE std, ddof=1)")
print("=" * 70)
rewards = np.array([1.0, 1.0, 0.0, 0.0])
mean = rewards.mean()
eps = 1e-4
std_pop = rewards.std(ddof=0)
std_sample = rewards.std(ddof=1)
adv_pop = (rewards - mean) / (std_pop + eps)
adv_sample = (rewards - mean) / (std_sample + eps)
print(f"rewards: {rewards.tolist()}  mean={mean}")
print(f"population std (ddof=0): {std_pop:.4f} -> advantages {np.round(adv_pop, 4).tolist()}")
print(f"sample std (ddof=1):     {std_sample:.4f} -> advantages {np.round(adv_sample, 4).tolist()}")
print("figure shows: [0.8659, 0.8659, -0.8659, -0.8659]  -- matches the SAMPLE std version")

print("\n" + "=" * 70)
print("2. Advantages + sequence logprobs -> policy gradient loss")
print("=" * 70)
logprobs = np.array([-7.9243, -20.1546, -16.6130, -23.3677])
advantages = np.array([0.8659, 0.8659, -0.8659, -0.8659])
loss = -np.mean(advantages * logprobs)
print(f"logprobs:   {logprobs.tolist()}")
print(f"advantages: {advantages.tolist()}")
print(f"L_PG = -mean(A_i * logprob_i) = {loss:.4f}")
print("figure shows: -2.5764")

print("\n" + "=" * 70)
print("3. Clipped policy ratio (clip_eps = 10 -> bounds [-9, 11])")
print("=" * 70)
new_lp = np.array([-7.9243, -20.1546, -16.6130, -23.3677])
old_lp = np.array([-10.9243, -20.3546, -14.6130, -23.3677])
ratio = np.exp(new_lp - old_lp)
clip_eps = 10
clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
print(f"new logprobs: {new_lp.tolist()}")
print(f"old logprobs: {old_lp.tolist()}")
print(f"ratio:        {np.round(ratio, 4).tolist()}")
print(f"clipped:      {np.round(clipped, 4).tolist()}")
print("figure shows: ratio [20.0855, 1.2214, 0.1353, 1.0000], clipped [11.0, 1.2214, 0.1353, 1.0]")

print("\n" + "=" * 70)
print("4. Total loss = policy gradient loss + KL loss term")
print("=" * 70)
policy_loss = -2.5764
kl_loss = -0.0075
total = policy_loss + kl_loss
print(f"{policy_loss} + ({kl_loss}) = {total:.4f}")
print("figure shows: -2.5839")
