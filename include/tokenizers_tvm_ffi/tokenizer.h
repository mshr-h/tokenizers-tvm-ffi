#ifndef TOKENIZERS_TVM_FFI_TOKENIZER_H_
#define TOKENIZERS_TVM_FFI_TOKENIZER_H_

#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/container/array.h>
#include <tokenizers_cpp.h>

#include <memory>

namespace tokenizers_tvm_ffi {

namespace ffi = ::tvm::ffi;

class TokenizerObj : public ffi::Object {
 public:
  static void RegisterReflection();

  ffi::Array<int64_t> Encode(ffi::String text) const;
  ffi::Array<ffi::Array<int64_t>> EncodeBatch(ffi::Array<ffi::String> texts) const;
  ffi::String Decode(ffi::Array<int64_t> ids) const;
  int64_t VocabSize() const;
  ffi::String IdToToken(int64_t token_id) const;
  int64_t TokenToId(ffi::String token) const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tokenizers_tvm_ffi.Tokenizer", TokenizerObj, ffi::Object);

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  friend class Tokenizer;
};

class Tokenizer : public ffi::ObjectRef {
 public:
  static Tokenizer FromJsonBytes(ffi::Bytes blob);
  static Tokenizer FromSentencePieceBytes(ffi::Bytes blob);
  static Tokenizer FromByteLevelBPE(ffi::Bytes vocab_blob, ffi::Bytes merges_blob,
                                    ffi::Bytes added_tokens_blob);
  static Tokenizer FromRWKVWorldPath(ffi::String path);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tokenizer, ffi::ObjectRef, TokenizerObj);
};

}  // namespace tokenizers_tvm_ffi

#endif  // TOKENIZERS_TVM_FFI_TOKENIZER_H_
