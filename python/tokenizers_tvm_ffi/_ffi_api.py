"""FFI API bindings for tokenizers_tvm_ffi."""

# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import Object as _ffi_Object, register_object as _FFI_REG_OBJ
from tvm_ffi.libinfo import load_lib_module as _FFI_LOAD_LIB
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)
# tvm-ffi-stubgen(import-object): tvm_ffi.libinfo.load_lib_module;False;_FFI_LOAD_LIB
LIB = _FFI_LOAD_LIB("tokenizers_tvm_ffi", "tokenizers_tvm_ffi")
# tvm-ffi-stubgen(begin): global/tokenizers_tvm_ffi
# tvm-ffi-stubgen(end)
# tvm-ffi-stubgen(import-object): tvm_ffi.register_object;False;_FFI_REG_OBJ
# tvm-ffi-stubgen(import-object): ffi.Object;False;_ffi_Object
@_FFI_REG_OBJ("tokenizers_tvm_ffi.Tokenizer")
class Tokenizer(_ffi_Object):
    """FFI binding for `tokenizers_tvm_ffi.Tokenizer`."""

    # tvm-ffi-stubgen(begin): object/tokenizers_tvm_ffi.Tokenizer
    # fmt: off
    if TYPE_CHECKING:
        def __init__(self) -> None: ...
        @staticmethod
        def from_json_bytes(_0: bytes, /) -> Tokenizer: ...
        @staticmethod
        def from_sentencepiece_bytes(_0: bytes, /) -> Tokenizer: ...
        @staticmethod
        def from_byte_level_bpe(_0: bytes, _1: bytes, _2: bytes, /) -> Tokenizer: ...
        @staticmethod
        def from_rwkv_world_path(_0: str, /) -> Tokenizer: ...
        def encode(self, _1: str, /) -> Sequence[int]: ...
        def encode_batch(self, _1: Sequence[str], /) -> Sequence[Sequence[int]]: ...
        def decode(self, _1: Sequence[int], /) -> str: ...
        def vocab_size(self, /) -> int: ...
        def id_to_token(self, _1: int, /) -> str: ...
        def token_to_id(self, _1: str, /) -> int: ...
        @staticmethod
        def __c_ffi_init__(*args: Any) -> Any: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


__all__ = [
    # tvm-ffi-stubgen(begin): __all__
    "LIB",
    "Tokenizer",
    # tvm-ffi-stubgen(end)
]
