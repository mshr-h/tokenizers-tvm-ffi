"""Smoke tests for tokenizers_tvm_ffi."""

import json
import pathlib
import sys
from collections.abc import Sequence

import pytest

# The build output dir must be on PYTHONPATH or we use the python/ stub directory.
# Ensure the stub package is importable.
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tokenizers_tvm_ffi  # noqa: E402

SENTENCEPIECE_MODEL = (
    ROOT / "3rdparty" / "tokenizers-cpp" / "sentencepiece" / "python" / "test" / "test_model.model"
)


def test_import():
    """import tokenizers_tvm_ffi succeeds and exposes Tokenizer class."""
    assert hasattr(tokenizers_tvm_ffi, "Tokenizer")


# ---------------------------------------------------------------------------
# HF JSON tokenizer
# ---------------------------------------------------------------------------

_MINIMAL_HF_TOKENIZER_JSON = json.dumps(
    {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "[UNK]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2},
            "unk_token": "[UNK]",
        },
    }
).encode()


class TestFromJsonBytes:
    def test_create(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        assert tok is not None

    def test_encode_returns_list(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        ids = tok.encode("hello world")
        assert isinstance(ids, Sequence)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_known(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        ids = tok.encode("hello world")
        assert list(ids) == [1, 2]

    def test_decode_roundtrip(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        text = "hello world"
        ids = tok.encode(text)
        decoded = tok.decode(list(ids))
        assert decoded == text

    def test_vocab_size(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        assert tok.vocab_size() == 3

    def test_id_to_token(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        assert tok.id_to_token(1) == "hello"
        assert tok.id_to_token(2) == "world"

    def test_token_to_id(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        assert tok.token_to_id("hello") == 1
        assert tok.token_to_id("world") == 2

    def test_token_to_id_unknown(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        assert tok.token_to_id("nonexistent") == -1

    def test_id_to_token_roundtrip(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        ids = tok.encode("hello")
        token = tok.id_to_token(ids[0])
        assert tok.token_to_id(token) == ids[0]

    def test_encode_batch(self):
        tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(_MINIMAL_HF_TOKENIZER_JSON)
        batch = tok.encode_batch(["hello", "world"])
        assert len(batch) == 2
        assert list(batch[0]) == [1]
        assert list(batch[1]) == [2]


# ---------------------------------------------------------------------------
# SentencePiece tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SENTENCEPIECE_MODEL.exists(), reason="SentencePiece model not found")
class TestFromSentencePieceBytes:
    def test_create(self):
        blob = SENTENCEPIECE_MODEL.read_bytes()
        tok = tokenizers_tvm_ffi.Tokenizer.from_sentencepiece_bytes(blob)
        assert tok is not None

    def test_encode_decode(self):
        blob = SENTENCEPIECE_MODEL.read_bytes()
        tok = tokenizers_tvm_ffi.Tokenizer.from_sentencepiece_bytes(blob)
        text = "hello world"
        ids = tok.encode(text)
        assert len(ids) > 0
        decoded = tok.decode(list(ids))
        # SentencePiece may add/remove whitespace, so just check non-empty
        assert len(decoded) > 0

    def test_vocab_size_positive(self):
        blob = SENTENCEPIECE_MODEL.read_bytes()
        tok = tokenizers_tvm_ffi.Tokenizer.from_sentencepiece_bytes(blob)
        assert tok.vocab_size() > 0
