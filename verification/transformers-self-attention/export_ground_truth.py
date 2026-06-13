"""
Export ground truth values to ground_truth.json.

Run with:
  .venv/bin/python verification/transformers-self-attention/export_ground_truth.py

Regenerate any time reference.py changes. The JSON is embedded verbatim
in the HTML visualization's <script> block.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from reference import (
    EMBEDDINGS, W_Q, W_K, W_V, W_O,
    SEQ_LEN, D_MODEL, D_K, D_V, H, SEED, TOKENS,
    scaled_dot_product_attention, multi_head_attention,
)

def to_list(a): return a.tolist()

result = multi_head_attention(EMBEDDINGS, W_Q, W_K, W_V, W_O)

heads = []
for i, head in enumerate(result["heads"]):
    Q_i = EMBEDDINGS @ W_Q[i]
    K_i = EMBEDDINGS @ W_K[i]
    V_i = EMBEDDINGS @ W_V[i]
    heads.append({
        "Q":       to_list(Q_i),
        "K":       to_list(K_i),
        "V":       to_list(V_i),
        "scores":  to_list(head["scores"]),
        "scaled":  to_list(head["scaled"]),
        "weights": to_list(head["weights"]),
        "output":  to_list(head["output"]),
    })

ground_truth = {
    "metadata": {
        "description": "Ground truth for 2-head self-attention on 3-token toy example",
        "source":      "Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762",
        "tokens":      TOKENS,
        "d_model":     D_MODEL,
        "h":           H,
        "d_k":         D_K,
        "d_v":         D_V,
        "numpy_seed":  SEED,
        "tolerance":   1e-6,
    },
    "inputs": {
        "embeddings": to_list(EMBEDDINGS),
        "W_Q": [to_list(w) for w in W_Q],
        "W_K": [to_list(w) for w in W_K],
        "W_V": [to_list(w) for w in W_V],
        "W_O": to_list(W_O),
    },
    "heads": heads,
    "output": to_list(result["output"]),
}

out_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
with open(out_path, "w") as f:
    json.dump(ground_truth, f, indent=2)

print(f"Exported ground_truth.json ({os.path.getsize(out_path)} bytes)")
print(f"Tokens: {TOKENS}")
for i, h in enumerate(heads):
    w = np.array(h["weights"])
    print(f"Head {i} attention weights (each row sums to {w.sum(axis=-1).tolist()}):")
    print(np.array(h["weights"]).round(6))
