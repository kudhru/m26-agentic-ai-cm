"""
Verifies the foundational worked examples used at the start of the deck:
(1) standard next-token training loss = cross-entropy against a one-hot
    ground-truth distribution;
(2) standard (word-level) knowledge distillation = cross-entropy against
    the teacher's soft distribution instead;
(3) that soft-label loss equals forward KLD plus a teacher-only constant
    (H(teacher)), connecting directly to the deck's later forward-KLD slide.
"""
import numpy as np

vocab = ['the', 'cat', 'sat', 'dog', 'ran']
print("vocab:", vocab, " (0=the, 1=cat, 2=sat, 3=dog, 4=ran)")
print('context: "the cat ___"  (true next word: "sat")\n')

print("=" * 70)
print("1. Standard training: cross-entropy vs ONE-HOT ground truth")
print("=" * 70)
p_model = np.array([0.05, 0.10, 0.60, 0.05, 0.20])
y_true = np.array([0, 0, 1, 0, 0])  # one-hot for 'sat'
assert abs(p_model.sum() - 1) < 1e-9
print("model prediction p =", p_model.tolist())
print("ground truth y     =", y_true.tolist(), ' (one-hot on "sat")')
loss_terms = -y_true * np.log(p_model)
print("per-token terms -y_i*log(p_i):", np.round(loss_terms, 4).tolist())
CE_onehot = loss_terms.sum()
print(f"cross-entropy loss = -log(p_sat) = -log({p_model[2]}) = {CE_onehot:.4f}")
print('(only the TRUE token\'s probability matters -- every other p_i is ignored)')

print("\n" + "=" * 70)
print("2. Standard (word-level) KD: cross-entropy vs TEACHER's soft distribution")
print("=" * 70)
q_teacher = np.array([0.02, 0.05, 0.70, 0.03, 0.20])
assert abs(q_teacher.sum() - 1) < 1e-9
print("teacher distribution q =", q_teacher.tolist())
print('  (teacher also likes "ran" a bit -- "the cat ran" is plausible too)')
print("student prediction p   =", p_model.tolist(), " (same student as above)")
loss_terms_kd = -q_teacher * np.log(p_model)
print("per-token terms -q_i*log(p_i):", np.round(loss_terms_kd, 4).tolist())
CE_soft = loss_terms_kd.sum()
print(f"cross-entropy loss = {CE_soft:.4f}")
print(f"(compare to one-hot loss {CE_onehot:.4f} -- now EVERY token contributes)")

print("\n" + "=" * 70)
print("2b. Temperature scaling: softening the teacher (Hinton et al. 2015)")
print("=" * 70)
logits = np.array([1.0, 2.0, 4.5, 1.5, 3.2])  # illustrative teacher logits
for T in [1.0, 2.0, 4.0]:
    q_T = np.exp(logits / T) / np.sum(np.exp(logits / T))
    print(f"T={T}: q = {np.round(q_T, 4).tolist()}")
print("higher T spreads probability mass out -- reveals relative confidence")
print("across the WRONG answers too, which T=1 nearly hides")

print("\n" + "=" * 70)
print("3. The soft-label loss IS forward KLD, up to a teacher-only constant")
print("=" * 70)
H_teacher = -np.sum(q_teacher * np.log(q_teacher))
KL_fwd = np.sum(q_teacher * np.log(q_teacher / p_model))
print(f"H(teacher) = {H_teacher:.4f}  (fixed, does not depend on the student)")
print(f"KL[teacher||student] = {KL_fwd:.4f}")
print(f"H(teacher) + KL[teacher||student] = {H_teacher+KL_fwd:.4f}  (matches cross-entropy {CE_soft:.4f})")

print("\n" + "=" * 70)
print("3b. The loss floor: H(teacher), not 0, even for a PERFECT student")
print("=" * 70)
q_perfect = q_teacher.copy()  # student matches the teacher exactly
CE_perfect = -np.sum(q_teacher * np.log(q_perfect))
KL_perfect = np.sum(q_teacher * np.log(q_teacher / q_perfect))
print(f"student == teacher exactly: KL[p||q] = {KL_perfect:.4f}  (zero, as expected)")
print(f"  cross-entropy L_KD = {CE_perfect:.4f}  (equals H(teacher), NOT 0)")
print(f"  => a perfect student cannot drive L_KD below {H_teacher:.4f}")
