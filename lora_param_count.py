"""
LoRA trainable-parameter count, worked out with plain arithmetic (no numpy needed —
this is pure counting, but still scripted and printed rather than hand-computed, so
the numbers on the slide are verified, not guessed).

This script produces every number shown on the "LoRA: verified parameter savings"
slide in peft-compression-slides.html. Run it and compare its printed output to the
slide — they must match exactly.
"""


def full_finetune_params(d_in, d_out):
    """Every entry of the weight matrix W in R^(d_in x d_out) is trainable."""
    return d_in * d_out


def lora_params(d_in, d_out, r):
    """LoRA trains B in R^(d_in x r) and A in R^(r x d_out) instead of W."""
    return r * (d_in + d_out)


# ---------------------------------------------------------------------------
# 1. A single projection matrix, typical LLM hidden size.
# ---------------------------------------------------------------------------
d = 4096   # hidden size (e.g. LLaMA-7B)
r = 8      # LoRA rank

full_one = full_finetune_params(d, d)
lora_one = lora_params(d, d, r)

print(f"Single {d}x{d} matrix, LoRA rank r={r}:")
print(f"  full fine-tuning params = d * d           = {full_one:,}")
print(f"  LoRA params             = r * (d + d)      = {lora_one:,}")
print(f"  reduction factor        = {full_one / lora_one:,.1f}x")

# ---------------------------------------------------------------------------
# 2. Every attention projection matrix (W_Q, W_K, W_V, W_O) in every layer of
#    a LLaMA-7B-shaped model.
# ---------------------------------------------------------------------------
n_layers = 32
n_matrices_per_layer = 4  # W_Q, W_K, W_V, W_O
total_model_params = 6_738_415_616  # LLaMA-7B's actual published parameter count

full_all = n_layers * n_matrices_per_layer * full_one
lora_all = n_layers * n_matrices_per_layer * lora_one

print(f"\nAll attention projections, {n_layers} layers x {n_matrices_per_layer} matrices:")
print(f"  full fine-tuning params = {full_all:,}")
print(f"  LoRA params             = {lora_all:,}")
print(f"  reduction factor        = {full_all / lora_all:,.1f}x")

pct_of_model = 100 * lora_all / total_model_params
print(f"\nLoRA params as a fraction of the {total_model_params:,}-param model:")
print(f"  {lora_all:,} / {total_model_params:,} = {pct_of_model:.3f}% trainable")
