"""
Structural tests for real_attention.json extracted from DistilBERT.

Verifies that every attention matrix is a valid probability distribution
and that the JSON shape is consistent (layers, heads, tokens).

These tests do NOT check exact values — they check the mathematical
properties that must hold for any valid attention output.
"""

import json
import os
import pytest

JSON_PATH = os.path.join(os.path.dirname(__file__), "real_attention.json")
TOL = 0.01  # rounding tolerance (values stored at 4 d.p.)


@pytest.fixture(scope="module")
def data():
    assert os.path.exists(JSON_PATH), (
        f"real_attention.json not found at {JSON_PATH}. "
        "Run extract_real_attention.py first."
    )
    with open(JSON_PATH) as f:
        return json.load(f)


# ===========================================================================
# Metadata
# ===========================================================================

class TestRealAttentionMetadata:
    def test_has_metadata(self, data):
        assert "metadata" in data

    def test_model_is_distilbert(self, data):
        assert "distilbert" in data["metadata"]["model"].lower()

    def test_six_layers(self, data):
        assert data["metadata"]["n_layers"] == 6

    def test_twelve_heads(self, data):
        assert data["metadata"]["n_heads"] == 12

    def test_five_sentences(self, data):
        assert len(data["sentences"]) == 5

    def test_each_sentence_has_required_fields(self, data):
        for s in data["sentences"]:
            for field in ("text", "note", "tokens", "attention"):
                assert field in s, f"Sentence '{s.get('text','?')}' missing '{field}'"


# ===========================================================================
# Shape consistency
# ===========================================================================

class TestRealAttentionShape:
    def test_each_sentence_has_six_layers(self, data):
        n_layers = data["metadata"]["n_layers"]
        for s in data["sentences"]:
            assert len(s["attention"]) == n_layers, (
                f"'{s['text']}': got {len(s['attention'])} layers, expected {n_layers}"
            )

    def test_each_layer_has_twelve_heads(self, data):
        n_heads = data["metadata"]["n_heads"]
        for s in data["sentences"]:
            for li, layer in enumerate(s["attention"]):
                assert len(layer) == n_heads, (
                    f"'{s['text']}' layer {li}: got {len(layer)} heads, expected {n_heads}"
                )

    def test_attention_matrix_is_square_and_matches_token_count(self, data):
        for s in data["sentences"]:
            n = len(s["tokens"])
            for li, layer in enumerate(s["attention"]):
                for hi, head in enumerate(layer):
                    assert len(head) == n, (
                        f"'{s['text']}' [{li},{hi}]: {len(head)} rows, expected {n}"
                    )
                    for ri, row in enumerate(head):
                        assert len(row) == n, (
                            f"'{s['text']}' [{li},{hi}] row {ri}: {len(row)} cols, expected {n}"
                        )


# ===========================================================================
# Probability distribution properties
# ===========================================================================

class TestRealAttentionDistribution:
    def test_rows_sum_to_one(self, data):
        for s in data["sentences"]:
            for li, layer in enumerate(s["attention"]):
                for hi, head in enumerate(layer):
                    for ri, row in enumerate(head):
                        row_sum = sum(row)
                        assert abs(row_sum - 1.0) < TOL, (
                            f"'{s['text']}' [{li},{hi}] row {ri}: "
                            f"sum={row_sum:.5f}, expected 1.0"
                        )

    def test_weights_non_negative(self, data):
        for s in data["sentences"]:
            for li, layer in enumerate(s["attention"]):
                for hi, head in enumerate(layer):
                    for ri, row in enumerate(head):
                        for ci, v in enumerate(row):
                            assert v >= -TOL, (
                                f"'{s['text']}' [{li},{hi}][{ri},{ci}]: "
                                f"negative weight {v:.6f}"
                            )

    def test_weights_at_most_one(self, data):
        for s in data["sentences"]:
            for li, layer in enumerate(s["attention"]):
                for hi, head in enumerate(layer):
                    for ri, row in enumerate(head):
                        for ci, v in enumerate(row):
                            assert v <= 1.0 + TOL, (
                                f"'{s['text']}' [{li},{hi}][{ri},{ci}]: "
                                f"weight > 1.0: {v:.6f}"
                            )


# ===========================================================================
# Token sanity
# ===========================================================================

class TestRealAttentionTokens:
    def test_tokens_start_with_cls(self, data):
        for s in data["sentences"]:
            assert s["tokens"][0] == "[CLS]", (
                f"'{s['text']}': first token is {s['tokens'][0]!r}, expected '[CLS]'"
            )

    def test_tokens_end_with_sep(self, data):
        for s in data["sentences"]:
            assert s["tokens"][-1] == "[SEP]", (
                f"'{s['text']}': last token is {s['tokens'][-1]!r}, expected '[SEP]'"
            )

    def test_sequence_length_reasonable(self, data):
        """Sequence length should be between 4 and 30 for our preset sentences."""
        for s in data["sentences"]:
            n = len(s["tokens"])
            assert 4 <= n <= 30, (
                f"'{s['text']}': unexpected token count {n}"
            )
