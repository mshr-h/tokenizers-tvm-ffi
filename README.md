# tokenizers-tvm-ffi

TVM-FFI wrapper for [mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp).

Exposes HuggingFace JSON, SentencePiece, ByteLevel BPE, and RWKV World tokenizers as a TVM-FFI class usable from Python.

## Install via pip

```bash
uv pip install git+https://github.com/mshr-h/tokenizers-tvm-ffi.git
```

## Usage

```python
import tokenizers_tvm_ffi

tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(
    open("tokenizer.json", "rb").read()
)
ids = tok.encode("hello world")
text = tok.decode(ids)
size = tok.vocab_size()
token = tok.id_to_token(ids[0])
tid = tok.token_to_id(token)
```

### Factory Methods

| Method | Input |
|--------|-------|
| `Tokenizer.from_json_bytes(blob)` | HuggingFace `tokenizer.json` as `bytes` |
| `Tokenizer.from_sentencepiece_bytes(blob)` | SentencePiece `.model` as `bytes` |
| `Tokenizer.from_byte_level_bpe(vocab, merges, added_tokens)` | BPE vocab/merges/added_tokens as `bytes` |
| `Tokenizer.from_rwkv_world_path(path)` | Filesystem path to RWKV World model as `str` |

### Instance Methods

| Method | Description |
|--------|-------------|
| `encode(text) -> Sequence[int]` | Encode text into token ids |
| `encode_batch(texts) -> Sequence[Sequence[int]]` | Batch encode |
| `decode(ids) -> str` | Decode token ids into text |
| `vocab_size() -> int` | Vocabulary size |
| `id_to_token(id) -> str` | Map token id to string |
| `token_to_id(token) -> int` | Map token string to id (-1 if not found) |


## Build from source

### Prerequisites

- C++17 compiler
- Rust / Cargo (for the tokenizers-cpp Rust backend)
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
git submodule update --init --recursive
uv sync
```

### Build

```bash
uv run cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev
uv run cmake --build build -j$(nproc)
```

## Test

```bash
uv run pytest -v
```

## Lint / Format

```bash
uv run ruff format .
uv run ruff check .
```

## License

See [LICENSE](LICENSE).
