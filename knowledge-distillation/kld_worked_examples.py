"""
Verifies the small numeric illustrations used on the forward-vs-reverse
KLD slides of the MiniLLM deck: (1) KL divergence is asymmetric -- KL[p||q]
and KL[q||p] are different numbers for the same p, q; (2) the mechanism
behind mode-covering (forward) vs mode-seeking (reverse) behavior -- which
kind of mismatch each direction penalizes heavily vs barely; (3) a small
discrete-vocabulary toy example showing forward KLD forcing a
capacity-limited student to place real probability on tokens the teacher
considers implausible.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def kl(a, b):
    return np.sum(a * np.log(a / b))


print("=" * 70)
print("1. KL divergence is asymmetric")
print("=" * 70)
p = np.array([0.70, 0.25, 0.05])
q = np.array([0.40, 0.40, 0.20])
assert abs(p.sum() - 1) < 1e-9 and abs(q.sum() - 1) < 1e-9
print("p =", p.tolist())
print("q =", q.tolist())

fwd = kl(p, q)
print("\nforward KLD, KL[p||q] = sum p*log(p/q):")
for i in range(3):
    print(f"  y={i+1}: {p[i]:.2f} * log({p[i]:.2f}/{q[i]:.2f}) = {p[i]*np.log(p[i]/q[i]):.4f}")
print(f"  KL[p||q] = {fwd:.4f}")

rev = kl(q, p)
print("\nreverse KLD, KL[q||p] = sum q*log(q/p):")
for i in range(3):
    print(f"  y={i+1}: {q[i]:.2f} * log({q[i]:.2f}/{p[i]:.2f}) = {q[i]*np.log(q[i]/p[i]):.4f}")
print(f"  KL[q||p] = {rev:.4f}")
print(f"\nKL[p||q] != KL[q||p]: {fwd:.4f} vs {rev:.4f} -- KL divergence is asymmetric")

print("\n" + "=" * 70)
print("2. Why: which mismatch each direction penalizes")
print("=" * 70)
print("Case A: q spends probability where p is tiny (q=0.20, p=0.01)")
qa, pa = 0.20, 0.01
print(f"  reverse-KL term  q*log(q/p) = {qa}*log({qa}/{pa}) = {qa*np.log(qa/pa):.4f}  <- big penalty")
print(f"  forward-KL term  p*log(p/q) = {pa}*log({pa}/{qa}) = {pa*np.log(pa/qa):.4f}  <- tiny")

print("\nCase B: q ignores a region p likes (q=0.001, p=0.30)")
qb, pb = 0.001, 0.30
print(f"  forward-KL term  p*log(p/q) = {pb}*log({pb}/{qb}) = {pb*np.log(pb/qb):.4f}  <- big penalty")
print(f"  reverse-KL term  q*log(q/p) = {qb}*log({qb}/{pb}) = {qb*np.log(qb/pb):.4f}  <- tiny")

print("\n" + "=" * 70)
print("3. A small-vocabulary toy example: forward KLD spills into the void")
print("=" * 70)
# context: "The cat ___"  -- teacher genuinely likes two DIFFERENT continuations
# ("sat", "ran") and almost never "purred" or "flew". Arrange the 4 tokens on a
# line with the two real modes at the ends and the implausible tokens in the
# middle, so a capacity-limited student -- restricted to a single symmetric
# bump over this line, e.g. because it can only represent one coherent
# "direction" of continuation -- must choose where to center its one bump.
tokens = ['sat', 'purred', 'flew', 'ran']
pos = np.array([0, 1, 2, 3])
p_vocab = np.array([0.49, 0.01, 0.01, 0.49])
assert abs(p_vocab.sum() - 1) < 1e-9
print('context: "the cat ___"')
print("tokens (by position):", tokens)
print("teacher p:           ", p_vocab.tolist())
print('  ("sat"/"ran" are the two real modes; "purred"/"flew" are near-impossible)')


def student(mu, sigma=0.6):
    logits = -(pos - mu) ** 2 / (2 * sigma ** 2)
    e = np.exp(logits - logits.max())
    return e / e.sum()


mus = np.linspace(0, 3, 601)
fwd_vals = [kl(p_vocab, student(mu)) for mu in mus]
rev_vals = [kl(student(mu), p_vocab) for mu in mus]
mu_fwd = mus[int(np.argmin(fwd_vals))]
mu_rev = mus[int(np.argmin(rev_vals))]
q_fwd = student(mu_fwd)
q_rev = student(mu_rev)

print(f"\nstudent family: single Gaussian-shaped bump over the 4 positions, center mu")
print(f"forward-KLD-optimal mu = {mu_fwd:.2f}  ->  q = {q_fwd.tolist()}")
print(f"  probability student puts on the two VOID tokens (purred+flew): {q_fwd[1]+q_fwd[2]:.4f}")
print(f"  (teacher rates those two tokens at only {p_vocab[1]+p_vocab[2]:.2f} combined)")
print(f"\nreverse-KLD-optimal mu = {mu_rev:.2f}  ->  q = {q_rev.tolist()}")
print(f"  probability student puts on the two VOID tokens (purred+flew): {q_rev[1]+q_rev[2]:.4f}")
print(f"\n=> minimizing forward KLD drives the student to straddle both modes,")
print(f"   spending {q_fwd[1]+q_fwd[2]:.0%} probability on tokens the teacher almost never uses;")
print(f"   minimizing reverse KLD instead collapses onto one real mode ('{tokens[int(round(mu_rev))]}'),")
print(f"   spending far less on the void ({q_rev[1]+q_rev[2]:.0%}).")
