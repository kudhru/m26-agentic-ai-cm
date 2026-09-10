"""
Verifies every numeric example used in the decoding-and-sampling deck:
(1) greedy decoding on a small 2-step vocabulary, showing it can miss the
    overall best sequence; (2) beam search (width 2) on the same vocabulary,
    showing it recovers the sequence greedy misses, and that beam width 1
    reduces to greedy; (3) length normalization (Wu et al., 2016) changing
    how a short vs. a longer sequence compare; (4) temperature scaling;
    (5) top-k sampling (Fan et al., 2018) on a peaked and a flat distribution;
    (6) nucleus / top-p sampling (Holtzman et al., 2020) on the same two
    distributions; (7) min-p sampling (Nguyen et al., 2024) on the same two;
    (8) the CTRL repetition penalty (Keskar et al., 2019).
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ======================================================================
# 1. Greedy decoding: a small 2-step example where it misses the best
#    overall sequence
# ======================================================================
print("=" * 70)
print("1. Greedy decoding: 'The weather today is ___'")
print("=" * 70)
step1 = {"bright": 0.50, "sunny": 0.35, "cold": 0.15}
step2_given = {
    "bright": {"outside": 0.50, "today": 0.30, "again": 0.20},
    "sunny": {"outside": 0.95, "today": 0.03, "again": 0.02},
    "cold": {"outside": 0.60, "today": 0.25, "again": 0.15},
}
for k, v in step1.items():
    assert abs(sum(step2_given[k].values()) - 1.0) < 1e-9
assert abs(sum(step1.values()) - 1.0) < 1e-9

print("step 1 distribution:", step1)
greedy1 = max(step1, key=step1.get)
print(f"greedy picks step 1 = '{greedy1}' (p={step1[greedy1]})")
greedy2 = max(step2_given[greedy1], key=step2_given[greedy1].get)
p_greedy = step1[greedy1] * step2_given[greedy1][greedy2]
print(f"greedy picks step 2 = '{greedy2}' (p={step2_given[greedy1][greedy2]})")
print(f"greedy sequence: '{greedy1} {greedy2}', P = {step1[greedy1]}*{step2_given[greedy1][greedy2]} = {p_greedy:.4f}")

print("\nEvery full 2-token sequence and its probability:")
all_seqs = {}
for w1, p1 in step1.items():
    for w2, p2 in step2_given[w1].items():
        all_seqs[(w1, w2)] = p1 * p2
        print(f"  '{w1} {w2}': {p1}*{p2} = {p1*p2:.4f}")
total = sum(all_seqs.values())
print(f"sum over all sequences = {total:.4f} (sanity check, should be 1.0)")
best_seq = max(all_seqs, key=all_seqs.get)
print(f"\nBEST overall sequence: '{best_seq[0]} {best_seq[1]}' = {all_seqs[best_seq]:.4f}")
print(f"greedy's sequence probability = {p_greedy:.4f}")
print(f"=> greedy is myopic: it never finds '{best_seq[0]} {best_seq[1]}' because "
      f"'{best_seq[0]}' did not have the single highest step-1 probability")

# ======================================================================
# 2. Beam search (width 2) on the exact same distributions
# ======================================================================
print("\n" + "=" * 70)
print("2. Beam search (width B=2) on the same distributions")
print("=" * 70)
B = 2
beam1 = sorted(step1.items(), key=lambda kv: -kv[1])[:B]
pruned = [w for w in step1 if w not in dict(beam1)]
print(f"step 1: keep top {B} of {len(step1)} -> {beam1}  (pruned: {pruned})")

candidates = []
for w1, p1 in beam1:
    for w2, p2 in step2_given[w1].items():
        candidates.append(((w1, w2), p1 * p2))
print(f"\nexpand each kept beam -> {len(candidates)} candidates:")
for seq, p in sorted(candidates, key=lambda c: -c[1]):
    print(f"  '{seq[0]} {seq[1]}': {p:.4f}")

beam2 = sorted(candidates, key=lambda c: -c[1])[:B]
print(f"\nstep 2: keep top {B} of {len(candidates)} -> final beam:")
for seq, p in beam2:
    print(f"  '{seq[0]} {seq[1]}': {p:.4f}")
best_beam = beam2[0]
print(f"\nbeam search's best sequence: '{best_beam[0][0]} {best_beam[0][1]}' = {best_beam[1]:.4f}")
print(f"greedy's sequence:            '{greedy1} {greedy2}' = {p_greedy:.4f}")
print(f"=> beam search (B={B}) finds the true best sequence ('{best_seq[0]} {best_seq[1]}' "
      f"= {all_seqs[best_seq]:.4f}) that greedy missed")

print("\nSanity check: beam width B=1 must reduce to plain greedy")
B1_beam1 = sorted(step1.items(), key=lambda kv: -kv[1])[:1]
B1_cands = []
for w1, p1 in B1_beam1:
    for w2, p2 in step2_given[w1].items():
        B1_cands.append(((w1, w2), p1 * p2))
B1_final = sorted(B1_cands, key=lambda c: -c[1])[:1][0]
print(f"  B=1 result: '{B1_final[0][0]} {B1_final[0][1]}' = {B1_final[1]:.4f}  "
      f"(matches greedy: {B1_final[1] == p_greedy})")

# ======================================================================
# 3. Length normalization (Wu et al., 2016 GNMT), Eq. 14: lp(Y)=(5+|Y|)^a/(5+1)^a
# ======================================================================
print("\n" + "=" * 70)
print("3. Length normalization: a longer, more confident-per-token sequence")
print("   vs. a shorter one -- which does raw log-probability favor?")
print("=" * 70)
# Two independently-decoded candidate sequences of different length.
seqA_tokens = [0.60, 0.60]                    # candidate A: 2 tokens
seqB_tokens = [0.75, 0.75, 0.75, 0.75]        # candidate B: 4 tokens, higher per-token confidence
p_A = float(np.prod(seqA_tokens))
p_B = float(np.prod(seqB_tokens))
logp_A, logp_B = np.log(p_A), np.log(p_B)
print(f"candidate A ({len(seqA_tokens)} tokens): per-token probs {seqA_tokens}, "
      f"P={p_A:.4f}, log P = {logp_A:.4f}, avg per-token prob = {p_A**(1/len(seqA_tokens)):.4f}")
print(f"candidate B ({len(seqB_tokens)} tokens): per-token probs {seqB_tokens}, "
      f"P={p_B:.4f}, log P = {logp_B:.4f}, avg per-token prob = {p_B**(1/len(seqB_tokens)):.4f}")
print(f"raw log-prob prefers A ({logp_A:.4f} > {logp_B:.4f}) even though B is MORE confident "
      f"per token on average -- B just accumulates more negative-log terms for having more of them")

alpha = 0.6  # Wu et al. report alpha in [0.6, 0.7] works best
def lp(length, a=alpha):
    return (5 + length) ** a / (5 + 1) ** a

lpA, lpB = lp(len(seqA_tokens)), lp(len(seqB_tokens))
score_A = logp_A / lpA
score_B = logp_B / lpB
print(f"\nlength penalty lp(Y)=(5+|Y|)^{alpha}/(5+1)^{alpha}: "
      f"lp({len(seqA_tokens)})={lpA:.4f}, lp({len(seqB_tokens)})={lpB:.4f}")
print(f"normalized score(A) = {logp_A:.4f}/{lpA:.4f} = {score_A:.4f}")
print(f"normalized score(B) = {logp_B:.4f}/{lpB:.4f} = {score_B:.4f}")
winner_raw = "A" if logp_A > logp_B else "B"
winner_norm = "A" if score_A > score_B else "B"
print(f"=> raw log-prob picks {winner_raw}; length-normalized score picks {winner_norm} "
      f"-- normalizing by length lets B's higher per-token confidence outweigh simply being longer")

# ======================================================================
# 4. Temperature scaling on a PEAKED distribution
# ======================================================================
print("\n" + "=" * 70)
print("4. Temperature scaling: '...capital of ___'")
print("=" * 70)
tokens_peaked = ["France", "Europe", "the", "a", "country", "Spain"]
logits_peaked = np.array([3.0, 0.5, 0.3, 0.0, -0.5, -1.5])

def softmax_T(logits, T):
    z = logits / T
    e = np.exp(z - z.max())
    return e / e.sum()

for T in [0.5, 1.0, 2.0]:
    probs = softmax_T(logits_peaked, T)
    print(f"T={T}: " + ", ".join(f"{t}={p:.4f}" for t, p in zip(tokens_peaked, probs)))
probs_peaked = softmax_T(logits_peaked, 1.0)

# ======================================================================
# 5-7. Top-k / top-p / min-p on a PEAKED and a FLAT distribution
# ======================================================================
print("\n" + "=" * 70)
print("5-7. Top-k, top-p, and min-p on a PEAKED vs. a FLAT distribution")
print("=" * 70)
tokens_flat = ["blue", "red", "green", "yellow", "purple", "orange"]
logits_flat = np.array([0.30, 0.20, 0.10, 0.00, -0.10, -0.50])
probs_flat = softmax_T(logits_flat, 1.0)

print("PEAKED distribution ('...capital of ___', T=1):")
for t, p in zip(tokens_peaked, probs_peaked):
    print(f"  {t:8s} {p:.4f}")
print("\nFLAT distribution ('My favorite color is ___', T=1):")
for t, p in zip(tokens_flat, probs_flat):
    print(f"  {t:8s} {p:.4f}")


def top_k(tokens, probs, k):
    order = np.argsort(-probs)
    kept = order[:k]
    kept_probs = probs[kept]
    renorm = kept_probs / kept_probs.sum()
    return [(tokens[i], probs[i], r) for i, r in zip(kept, renorm)]


def top_p(tokens, probs, p):
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs)
    n_keep = int(np.searchsorted(cum, p) + 1)
    kept = order[:n_keep]
    kept_probs = probs[kept]
    renorm = kept_probs / kept_probs.sum()
    return [(tokens[i], probs[i], r) for i, r in zip(kept, renorm)], n_keep, cum[n_keep - 1]


def min_p(tokens, probs, p_base):
    p_max = probs.max()
    p_scaled = p_base * p_max
    kept = np.where(probs >= p_scaled)[0]
    kept_probs = probs[kept]
    renorm = kept_probs / kept_probs.sum()
    return [(tokens[i], probs[i], r) for i, r in zip(kept, renorm)], p_scaled


print("\n--- top-k, k=3 ---")
for name, tokens, probs in [("PEAKED", tokens_peaked, probs_peaked), ("FLAT", tokens_flat, probs_flat)]:
    kept = top_k(tokens, probs, 3)
    print(f"{name}: kept={[t for t,_,_ in kept]}")
    for t, orig, renorm in kept:
        print(f"  {t:8s} orig={orig:.4f} -> renormalized={renorm:.4f}")

print("\n--- top-p (nucleus), p=0.90 ---")
for name, tokens, probs in [("PEAKED", tokens_peaked, probs_peaked), ("FLAT", tokens_flat, probs_flat)]:
    kept, n_keep, cum_reached = top_p(tokens, probs, 0.90)
    print(f"{name}: kept {n_keep} tokens to reach cumulative {cum_reached:.4f} >= 0.90 -> {[t for t,_,_ in kept]}")
    for t, orig, renorm in kept:
        print(f"  {t:8s} orig={orig:.4f} -> renormalized={renorm:.4f}")

print("\n--- min-p, p_base=0.10 ---")
for name, tokens, probs in [("PEAKED", tokens_peaked, probs_peaked), ("FLAT", tokens_flat, probs_flat)]:
    kept, p_scaled = min_p(tokens, probs, 0.10)
    print(f"{name}: p_max={probs.max():.4f}, p_scaled=0.10*{probs.max():.4f}={p_scaled:.4f} -> kept {[t for t,_,_ in kept]}")
    for t, orig, renorm in kept:
        print(f"  {t:8s} orig={orig:.4f} -> renormalized={renorm:.4f}")

# ======================================================================
# 8. Repetition penalty (Keskar et al., 2019, CTRL)
# ======================================================================
print("\n" + "=" * 70)
print("8. Repetition penalty (CTRL, Keskar et al. 2019)")
print("=" * 70)
tokens_rep = ["the", "cat", "sat", "again"]
logits_rep = np.array([1.2, 0.8, 2.0, 0.3])
already_generated = {"cat"}  # 'cat' appeared earlier in the sequence
theta = 1.5
print(f"logits: {dict(zip(tokens_rep, logits_rep))}")
print(f"already generated: {already_generated}, theta={theta}")

probs_before = softmax_T(logits_rep, 1.0)
adj_logits = np.array([
    z / theta if t in already_generated else z
    for t, z in zip(tokens_rep, logits_rep)
])
probs_after = softmax_T(adj_logits, 1.0)
print(f"logits after penalty: {dict(zip(tokens_rep, np.round(adj_logits,4)))}")
print("token      P(before)   P(after)")
for t, b, a in zip(tokens_rep, probs_before, probs_after):
    print(f"  {t:8s} {b:.4f}      {a:.4f}")
print(f"=> P('cat') drops from {probs_before[1]:.4f} to {probs_after[1]:.4f} after dividing its logit by theta={theta}")
