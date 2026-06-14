"""
Extract real attention weights from DistilBERT for preset sentences.

Outputs: real_attention.json — attention[layer][head][i][j] = weight from
token i to token j, post-softmax, rows sum to 1.

Run with:
  .venv/bin/python verification/transformers-self-attention/extract_real_attention.py
"""

import json
import sys
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# ---------------------------------------------------------------------------
# Preset sentences — chosen to showcase distinct attention phenomena
# ---------------------------------------------------------------------------
PRESET_SENTENCES = [
    {
        "text": "The cat sat on the mat",
        "note": "Simple baseline. Early layers show local/positional patterns; "
                "later layers show more global structure.",
    },
    {
        "text": "The animal did not cross the street because it was too tired",
        "note": "Coreference: watch 'it' — in trained models it attends "
                "strongly to 'animal', not 'street'. Classic example from "
                "Alammar (2018) 'The Illustrated Transformer'.",
    },
    {
        "text": "She loves New York but he hates it",
        "note": "Multiple pronouns. 'it' refers to 'New York'; 'she' and 'he' "
                "contrast. Look for symmetric or anti-correlated patterns.",
    },
    {
        "text": "The bank by the river flooded yesterday",
        "note": "Lexical disambiguation: 'bank' is the riverbank, not financial. "
                "Watch whether 'bank' attends more to 'river' or to nothing special.",
    },
    {
        "text": "John told Mary that he would leave soon",
        "note": "Nested coreference: 'he' refers to 'John'. Some heads in "
                "middle layers pick this up.",
    },
]

MODEL_NAME = "distilbert-base-uncased"


def extract():
    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertModel.from_pretrained(MODEL_NAME, output_attentions=True)
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers   # 6 for DistilBERT
    n_heads  = cfg.num_attention_heads  # 12 for DistilBERT

    sentences_out = []
    for item in PRESET_SENTENCES:
        text = item["text"]
        print(f"  Processing: {text!r}", flush=True)

        inputs   = tokenizer(text, return_tensors="pt")
        token_ids = inputs["input_ids"][0].tolist()
        tokens   = tokenizer.convert_ids_to_tokens(token_ids)

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.attentions: tuple of n_layers tensors
        # each tensor shape: (batch=1, n_heads, n_tokens, n_tokens)
        attention_by_layer = []
        for layer_idx, layer_att in enumerate(outputs.attentions):
            heads = []
            layer_np = layer_att[0].float().numpy()  # (n_heads, n_tokens, n_tokens)
            for head_att in layer_np:
                # Round to 4 decimal places — sufficient for color rendering
                row_list = [
                    [round(float(v), 4) for v in row]
                    for row in head_att
                ]
                heads.append(row_list)
            attention_by_layer.append(heads)

        sentences_out.append({
            "text":      text,
            "note":      item["note"],
            "tokens":    tokens,
            "attention": attention_by_layer,  # [layer][head][i][j]
        })

    output = {
        "metadata": {
            "model":       MODEL_NAME,
            "n_layers":    n_layers,
            "n_heads":     n_heads,
            "description": (
                "Post-softmax attention weights for all layers and heads. "
                "attention[layer][head][i][j] = weight query-i places on key-j. "
                "Rows sum to 1.0 (within float rounding at 4 d.p.)."
            ),
        },
        "sentences": sentences_out,
    }

    out_path = "verification/transformers-self-attention/real_attention.json"
    print(f"\nWriting {out_path}...", flush=True)
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    # ---------------------------------------------------------------------------
    # Validate every row sums to 1.0 within rounding tolerance
    # ---------------------------------------------------------------------------
    n_rows = 0
    bad_rows = 0
    for s in sentences_out:
        for layer in s["attention"]:
            for head in layer:
                for row in head:
                    n_rows += 1
                    row_sum = sum(row)
                    if abs(row_sum - 1.0) > 0.01:
                        bad_rows += 1
                        print(f"  WARNING: row sum = {row_sum:.5f}", flush=True)

    import os
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n{n_rows} attention rows validated ({bad_rows} bad).")
    print(f"Output size: {size_kb:.1f} KB")
    print("Done.")


if __name__ == "__main__":
    # Run from project root
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(root)
    extract()
