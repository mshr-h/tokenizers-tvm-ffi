#include "tokenizers_tvm_ffi/tokenizer.h"

#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace tokenizers_tvm_ffi {

namespace {

std::vector<int32_t> ArrayToVecI32(const ffi::Array<int64_t>& arr) {
  std::vector<int32_t> result;
  result.reserve(arr.size());
  for (size_t i = 0; i < arr.size(); ++i) {
    int64_t v = arr[i];
    if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max()) {
      TVM_FFI_THROW(ValueError) << "Token id " << v << " at index " << i
                                << " is out of int32 range";
    }
    result.push_back(static_cast<int32_t>(v));
  }
  return result;
}

ffi::Array<int64_t> VecI32ToArray(const std::vector<int32_t>& vec) {
  ffi::Array<int64_t> result;
  result.reserve(vec.size());
  for (int32_t v : vec) {
    result.push_back(static_cast<int64_t>(v));
  }
  return result;
}

}  // namespace

// --- Static factory methods ---

Tokenizer Tokenizer::FromJsonBytes(ffi::Bytes blob) {
  std::string data(blob.data(), blob.size());
  auto obj = ffi::make_object<TokenizerObj>();
  obj->tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(data);
  return Tokenizer(std::move(obj));
}

Tokenizer Tokenizer::FromSentencePieceBytes(ffi::Bytes blob) {
  std::string data(blob.data(), blob.size());
  auto obj = ffi::make_object<TokenizerObj>();
  obj->tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(data);
  return Tokenizer(std::move(obj));
}

Tokenizer Tokenizer::FromByteLevelBPE(ffi::Bytes vocab_blob, ffi::Bytes merges_blob,
                                      ffi::Bytes added_tokens_blob) {
  std::string vocab(vocab_blob.data(), vocab_blob.size());
  std::string merges(merges_blob.data(), merges_blob.size());
  std::string added(added_tokens_blob.data(), added_tokens_blob.size());
  auto obj = ffi::make_object<TokenizerObj>();
  obj->tokenizer_ = tokenizers::Tokenizer::FromBlobByteLevelBPE(vocab, merges, added);
  return Tokenizer(std::move(obj));
}

Tokenizer Tokenizer::FromRWKVWorldPath(ffi::String path) {
  std::string path_str(path.data(), path.size());
  // Read the file into a blob and use FromBlobRWKVWorld
  std::ifstream ifs(path_str, std::ios::binary);
  if (!ifs) {
    TVM_FFI_THROW(ValueError) << "Cannot open RWKV world model file: " << path_str;
  }
  std::string blob((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  auto obj = ffi::make_object<TokenizerObj>();
  obj->tokenizer_ = tokenizers::Tokenizer::FromBlobRWKVWorld(blob);
  return Tokenizer(std::move(obj));
}

// --- Instance methods ---

ffi::Array<int64_t> TokenizerObj::Encode(ffi::String text) const {
  std::string input(text.data(), text.size());
  std::vector<int32_t> ids = tokenizer_->Encode(input);
  return VecI32ToArray(ids);
}

ffi::Array<ffi::Array<int64_t>> TokenizerObj::EncodeBatch(ffi::Array<ffi::String> texts) const {
  std::vector<std::string> inputs;
  inputs.reserve(texts.size());
  for (size_t i = 0; i < texts.size(); ++i) {
    inputs.emplace_back(texts[i].data(), texts[i].size());
  }
  std::vector<std::vector<int32_t>> batch_ids = tokenizer_->EncodeBatch(inputs);
  ffi::Array<ffi::Array<int64_t>> result;
  result.reserve(batch_ids.size());
  for (const auto& ids : batch_ids) {
    result.push_back(VecI32ToArray(ids));
  }
  return result;
}

ffi::String TokenizerObj::Decode(ffi::Array<int64_t> ids) const {
  std::vector<int32_t> id_vec = ArrayToVecI32(ids);
  return ffi::String(tokenizer_->Decode(id_vec));
}

int64_t TokenizerObj::VocabSize() const {
  return static_cast<int64_t>(tokenizer_->GetVocabSize());
}

ffi::String TokenizerObj::IdToToken(int64_t token_id) const {
  if (token_id < std::numeric_limits<int32_t>::min() ||
      token_id > std::numeric_limits<int32_t>::max()) {
    TVM_FFI_THROW(ValueError) << "Token id " << token_id << " is out of int32 range";
  }
  return ffi::String(tokenizer_->IdToToken(static_cast<int32_t>(token_id)));
}

int64_t TokenizerObj::TokenToId(ffi::String token) const {
  std::string tok_str(token.data(), token.size());
  return static_cast<int64_t>(tokenizer_->TokenToId(tok_str));
}

// --- Reflection registration ---

void TokenizerObj::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TokenizerObj>()
      .def_static("from_json_bytes", &Tokenizer::FromJsonBytes,
                  "Load HF tokenizer.json from bytes.")
      .def_static("from_sentencepiece_bytes", &Tokenizer::FromSentencePieceBytes,
                  "Load SentencePiece model from bytes.")
      .def_static("from_byte_level_bpe", &Tokenizer::FromByteLevelBPE,
                  "Load ByteLevel BPE from vocab/merges bytes.")
      .def_static("from_rwkv_world_path", &Tokenizer::FromRWKVWorldPath,
                  "Load RWKVWorld tokenizer from filesystem path.")
      .def("encode", &TokenizerObj::Encode, "Encode text into token ids.")
      .def("encode_batch", &TokenizerObj::EncodeBatch,
           "Encode a batch of texts into token ids.")
      .def("decode", &TokenizerObj::Decode, "Decode token ids into text.")
      .def("vocab_size", &TokenizerObj::VocabSize, "Return vocabulary size.")
      .def("id_to_token", &TokenizerObj::IdToToken, "Map token id to token string.")
      .def("token_to_id", &TokenizerObj::TokenToId, "Map token string to token id.");
}

TVM_FFI_STATIC_INIT_BLOCK() { TokenizerObj::RegisterReflection(); }

}  // namespace tokenizers_tvm_ffi
