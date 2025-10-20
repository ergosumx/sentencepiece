// Copyright 2025 VecRax
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bindings/c/sentencepiece_c.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"

struct spc_sentencepiece_processor {
  sentencepiece::SentencePieceProcessor impl;
};

struct spc_sentencepiece_normalizer {
  sentencepiece::SentencePieceNormalizer impl;
};

namespace {

using sentencepiece::SentencePieceProcessor;
using sentencepiece::SentencePieceTrainer;
using sentencepiece::SentencePieceNormalizer;
using sentencepiece::util::Status;
using sentencepiece::util::StatusCode;

spc_status_t MakeStatus(spc_status_code_t code, const char *message) {
  spc_status_t status;
  status.code = code;
  if (message == nullptr) {
    status.message = nullptr;
    return status;
  }
  const size_t length = std::strlen(message);
  if (length == 0) {
    status.message = nullptr;
    return status;
  }
  char *buffer = static_cast<char *>(std::malloc(length + 1));
  if (buffer == nullptr) {
    status.message = nullptr;
    return status;
  }
  std::memcpy(buffer, message, length);
  buffer[length] = '\0';
  status.message = buffer;
  return status;
}

spc_status_t MakeStatusFromProto(const Status &status) {
  if (status.ok()) {
    return {SPC_STATUS_OK, nullptr};
  }
  const std::string message = status.ToString();
  return MakeStatus(static_cast<spc_status_code_t>(status.code()),
                    message.c_str());
}

spc_status_t MakeInternalError(const char *message) {
  return MakeStatus(SPC_STATUS_INTERNAL, message);
}

spc_status_t MakeInvalidArgument(const char *message) {
  return MakeStatus(SPC_STATUS_INVALID_ARGUMENT, message);
}

spc_status_t MakeResourceExhausted(const char *message) {
  return MakeStatus(SPC_STATUS_RESOURCE_EXHAUSTED, message);
}

bool RequiresProtoRewrite(const spc_encode_options_t &options) {
  return options.add_bos || options.add_eos || options.reverse ||
         options.emit_unk_piece;
}

spc_encode_options_t GetEncodeOptionsOrDefault(
    const spc_encode_options_t *options) {
  spc_encode_options_t result;
  if (options == nullptr) {
    spc_encode_options_init(&result);
    return result;
  }
  result = *options;
  return result;
}

spc_sample_encode_and_score_options_t GetSampleOptionsOrDefault(
    const spc_sample_encode_and_score_options_t *options) {
  spc_sample_encode_and_score_options_t result;
  if (options == nullptr) {
    spc_sample_encode_and_score_options_init(&result);
    return result;
  }
  result = *options;
  return result;
}

spc_encode_options_t FromSampleOptions(
    const spc_sample_encode_and_score_options_t &options) {
  spc_encode_options_t converted;
  converted.add_bos = options.add_bos;
  converted.add_eos = options.add_eos;
  converted.reverse = options.reverse;
  converted.emit_unk_piece = options.emit_unk_piece;
  converted.enable_sampling = false;
  converted.nbest_size = -1;
  converted.alpha = 0.0f;
  return converted;
}

absl::string_view ToStringView(spc_string_view_t view) {
  if (view.data == nullptr) {
    return absl::string_view();
  }
  return absl::string_view(view.data, view.length);
}

bool IsIdOutOfRange(int id, int piece_size) {
  return id < 0 || id >= piece_size;
}

void RewriteIds(const SentencePieceProcessor &processor,
                std::vector<int> *ids,
                const spc_encode_options_t &options) {
  if (!options.add_bos && !options.add_eos && !options.reverse) {
    return;
  }
  if (options.reverse) {
    std::reverse(ids->begin(), ids->end());
  }
  if (options.add_bos) {
    ids->insert(ids->begin(), processor.bos_id());
  }
  if (options.add_eos) {
    ids->push_back(processor.eos_id());
  }
}

void RewritePieces(const SentencePieceProcessor &processor,
                   std::vector<std::string> *pieces,
                   const spc_encode_options_t &options) {
  if (!options.add_bos && !options.add_eos && !options.reverse &&
      !options.emit_unk_piece) {
    return;
  }
  if (options.reverse) {
    std::reverse(pieces->begin(), pieces->end());
  }
  if (options.add_bos) {
    pieces->insert(pieces->begin(), processor.IdToPiece(processor.bos_id()));
  }
  if (options.add_eos) {
    pieces->push_back(processor.IdToPiece(processor.eos_id()));
  }
  if (options.emit_unk_piece) {
    const auto &unk = processor.IdToPiece(processor.unk_id());
    for (auto &piece : *pieces) {
      const int id = processor.PieceToId(piece);
      if (id == processor.unk_id()) {
        piece = unk;
      }
    }
  }
}

spc_status_t AssignBytes(const std::string &source, spc_bytes_t *dest) {
  if (dest == nullptr) {
    return MakeInvalidArgument("dest is null");
  }
  if (source.empty()) {
    dest->data = nullptr;
    dest->length = 0;
    return {SPC_STATUS_OK, nullptr};
  }
  char *buffer = static_cast<char *>(std::malloc(source.size()));
  if (buffer == nullptr) {
    return MakeResourceExhausted("memory allocation failed");
  }
  std::memcpy(buffer, source.data(), source.size());
  dest->data = buffer;
  dest->length = source.size();
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t AssignIntArray(const std::vector<int> &source,
                            spc_int_array_t *dest) {
  if (dest == nullptr) {
    return MakeInvalidArgument("dest is null");
  }
  if (source.empty()) {
    dest->data = nullptr;
    dest->length = 0;
    return {SPC_STATUS_OK, nullptr};
  }
  int32_t *buffer = static_cast<int32_t *>(
      std::malloc(sizeof(int32_t) * source.size()));
  if (buffer == nullptr) {
    return MakeResourceExhausted("memory allocation failed");
  }
  for (size_t i = 0; i < source.size(); ++i) {
    buffer[i] = static_cast<int32_t>(source[i]);
  }
  dest->data = buffer;
  dest->length = source.size();
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t AssignIntArrayList(const std::vector<std::vector<int>> &source,
                                spc_int_array_list_t *dest) {
  if (dest == nullptr) {
    return MakeInvalidArgument("dest is null");
  }
  dest->items = nullptr;
  dest->length = 0;
  if (source.empty()) {
    return {SPC_STATUS_OK, nullptr};
  }
  spc_int_array_t *list = static_cast<spc_int_array_t *>(
      std::calloc(source.size(), sizeof(spc_int_array_t)));
  if (list == nullptr) {
    return MakeResourceExhausted("memory allocation failed");
  }
  size_t index = 0;
  for (const auto &entry : source) {
    spc_status_t status = AssignIntArray(entry, &list[index]);
    if (!spc_status_is_ok(&status)) {
      for (size_t i = 0; i < index; ++i) {
        spc_int_array_destroy(&list[i]);
      }
      std::free(list);
      return status;
    }
    ++index;
  }
  dest->items = list;
  dest->length = source.size();
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t AssignBytesArray(const std::vector<std::string> &source,
                              spc_bytes_array_t *dest) {
  if (dest == nullptr) {
    return MakeInvalidArgument("dest is null");
  }
  dest->items = nullptr;
  dest->length = 0;
  if (source.empty()) {
    return {SPC_STATUS_OK, nullptr};
  }
  spc_bytes_t *items = static_cast<spc_bytes_t *>(
      std::calloc(source.size(), sizeof(spc_bytes_t)));
  if (items == nullptr) {
    return MakeResourceExhausted("memory allocation failed");
  }
  size_t index = 0;
  for (const auto &value : source) {
    spc_status_t status = AssignBytes(value, &items[index]);
    if (!spc_status_is_ok(&status)) {
      for (size_t i = 0; i < index; ++i) {
        spc_bytes_destroy(&items[i]);
      }
      std::free(items);
      return status;
    }
    ++index;
  }
  dest->items = items;
  dest->length = source.size();
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t AssignBytesArrayList(
    const std::vector<std::vector<std::string>> &source,
    spc_bytes_array_list_t *dest) {
  if (dest == nullptr) {
    return MakeInvalidArgument("dest is null");
  }
  dest->items = nullptr;
  dest->length = 0;
  if (source.empty()) {
    return {SPC_STATUS_OK, nullptr};
  }
  spc_bytes_array_t *items = static_cast<spc_bytes_array_t *>(
      std::calloc(source.size(), sizeof(spc_bytes_array_t)));
  if (items == nullptr) {
    return MakeResourceExhausted("memory allocation failed");
  }
  size_t index = 0;
  for (const auto &entry : source) {
    spc_status_t status = AssignBytesArray(entry, &items[index]);
    if (!spc_status_is_ok(&status)) {
      for (size_t i = 0; i < index; ++i) {
        spc_bytes_array_destroy(&items[i]);
      }
      std::free(items);
      return status;
    }
    ++index;
  }
  dest->items = items;
  dest->length = source.size();
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t AssignScoredIntArrayList(
    const std::vector<std::pair<std::vector<int>, float>> &source,
    spc_scored_int_array_list_t *dest) {
  if (dest == nullptr) {
    return MakeInvalidArgument("dest is null");
  }
  dest->items = nullptr;
  dest->length = 0;
  if (source.empty()) {
    return {SPC_STATUS_OK, nullptr};
  }
  spc_scored_int_array_t *items = static_cast<spc_scored_int_array_t *>(
      std::calloc(source.size(), sizeof(spc_scored_int_array_t)));
  if (items == nullptr) {
    return MakeResourceExhausted("memory allocation failed");
  }
  size_t index = 0;
  for (const auto &entry : source) {
    spc_status_t status = AssignIntArray(entry.first, &items[index].ids);
    if (!spc_status_is_ok(&status)) {
      for (size_t i = 0; i < index; ++i) {
        spc_int_array_destroy(&items[i].ids);
      }
      std::free(items);
      return status;
    }
    items[index].score = entry.second;
    ++index;
  }
  dest->items = items;
  dest->length = source.size();
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t AssignScoredBytesArrayList(
    const std::vector<std::pair<std::vector<std::string>, float>> &source,
    spc_scored_bytes_array_list_t *dest) {
  if (dest == nullptr) {
    return MakeInvalidArgument("dest is null");
  }
  dest->items = nullptr;
  dest->length = 0;
  if (source.empty()) {
    return {SPC_STATUS_OK, nullptr};
  }
  spc_scored_bytes_array_t *items = static_cast<spc_scored_bytes_array_t *>(
      std::calloc(source.size(), sizeof(spc_scored_bytes_array_t)));
  if (items == nullptr) {
    return MakeResourceExhausted("memory allocation failed");
  }
  size_t index = 0;
  for (const auto &entry : source) {
    spc_status_t status = AssignBytesArray(entry.first, &items[index].pieces);
    if (!spc_status_is_ok(&status)) {
      for (size_t i = 0; i < index; ++i) {
        spc_bytes_array_destroy(&items[i].pieces);
      }
      std::free(items);
      return status;
    }
    items[index].score = entry.second;
    ++index;
  }
  dest->items = items;
  dest->length = source.size();
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t AssignBytesArrayFlat(
    const std::vector<std::string> &source,
    spc_bytes_array_t *dest) {
  return AssignBytesArray(source, dest);
}

spc_status_t AssignFloatArray(const std::vector<float> &source,
                              spc_float_array_t *dest) {
  if (dest == nullptr) {
    return MakeInvalidArgument("dest is null");
  }
  dest->data = nullptr;
  dest->length = 0;
  if (source.empty()) {
    return {SPC_STATUS_OK, nullptr};
  }
  float *buffer = static_cast<float *>(
      std::malloc(sizeof(float) * source.size()));
  if (buffer == nullptr) {
    return MakeResourceExhausted("memory allocation failed");
  }
  std::memcpy(buffer, source.data(), sizeof(float) * source.size());
  dest->data = buffer;
  dest->length = source.size();
  return {SPC_STATUS_OK, nullptr};
}

template <typename Fn>
spc_status_t ParallelFor(size_t total, int32_t requested_threads, Fn fn) {
  if (total == 0) {
    return {SPC_STATUS_OK, nullptr};
  }
  int32_t threads = requested_threads;
  if (threads <= 0) {
    threads = static_cast<int32_t>(std::thread::hardware_concurrency());
  }
  if (threads <= 0) {
    threads = 1;
  }
  threads = std::max<int32_t>(1,
                              std::min<int32_t>(threads,
                                                 static_cast<int32_t>(total)));
  if (threads == 1) {
    for (size_t index = 0; index < total; ++index) {
      spc_status_t status = fn(index);
      if (!spc_status_is_ok(&status)) {
        return status;
      }
    }
    return {SPC_STATUS_OK, nullptr};
  }
  std::atomic<size_t> current{0};
  std::vector<std::thread> workers;
  std::mutex status_mutex;
  spc_status_t first_error = {SPC_STATUS_OK, nullptr};
  for (int32_t t = 0; t < threads; ++t) {
    workers.emplace_back([&]() {
      while (true) {
        const size_t index = current.fetch_add(1);
        if (index >= total) {
          break;
        }
        spc_status_t status = fn(index);
        if (!spc_status_is_ok(&status)) {
          std::lock_guard<std::mutex> lock(status_mutex);
          if (spc_status_is_ok(&first_error)) {
            first_error = status;
          } else {
            spc_status_destroy(&status);
          }
          break;
        }
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  if (!spc_status_is_ok(&first_error)) {
    return first_error;
  }
  return {SPC_STATUS_OK, nullptr};
}

}  // namespace

extern "C" {

// -----------------------------------------------------------------------------
// Status helpers
// -----------------------------------------------------------------------------

void spc_status_destroy(spc_status_t *status) {
  if (status == nullptr) {
    return;
  }
  if (status->message != nullptr) {
    std::free(status->message);
    status->message = nullptr;
  }
  status->code = SPC_STATUS_OK;
}

bool spc_status_is_ok(const spc_status_t *status) {
  return status != nullptr && status->code == SPC_STATUS_OK;
}

// -----------------------------------------------------------------------------
// Common value containers
// -----------------------------------------------------------------------------

spc_string_view_t spc_string_view_from_cstr(const char *value) {
  if (value == nullptr) {
    return {nullptr, 0};
  }
  return {value, std::strlen(value)};
}

void spc_bytes_destroy(spc_bytes_t *value) {
  if (value == nullptr) {
    return;
  }
  if (value->data != nullptr) {
    std::free(value->data);
  }
  value->data = nullptr;
  value->length = 0;
}

void spc_int_array_destroy(spc_int_array_t *value) {
  if (value == nullptr) {
    return;
  }
  if (value->data != nullptr) {
    std::free(value->data);
  }
  value->data = nullptr;
  value->length = 0;
}

void spc_float_array_destroy(spc_float_array_t *value) {
  if (value == nullptr) {
    return;
  }
  if (value->data != nullptr) {
    std::free(value->data);
  }
  value->data = nullptr;
  value->length = 0;
}

void spc_size_array_destroy(spc_size_array_t *value) {
  if (value == nullptr) {
    return;
  }
  if (value->data != nullptr) {
    std::free(value->data);
  }
  value->data = nullptr;
  value->length = 0;
}

void spc_int_array_list_destroy(spc_int_array_list_t *value) {
  if (value == nullptr) {
    return;
  }
  if (value->items != nullptr) {
    for (size_t i = 0; i < value->length; ++i) {
      spc_int_array_destroy(&value->items[i]);
    }
    std::free(value->items);
  }
  value->items = nullptr;
  value->length = 0;
}

void spc_bytes_array_destroy(spc_bytes_array_t *value) {
  if (value == nullptr) {
    return;
  }
  if (value->items != nullptr) {
    for (size_t i = 0; i < value->length; ++i) {
      spc_bytes_destroy(&value->items[i]);
    }
    std::free(value->items);
  }
  value->items = nullptr;
  value->length = 0;
}

void spc_bytes_array_list_destroy(spc_bytes_array_list_t *value) {
  if (value == nullptr) {
    return;
  }
  if (value->items != nullptr) {
    for (size_t i = 0; i < value->length; ++i) {
      spc_bytes_array_destroy(&value->items[i]);
    }
    std::free(value->items);
  }
  value->items = nullptr;
  value->length = 0;
}

void spc_scored_int_array_list_destroy(spc_scored_int_array_list_t *value) {
  if (value == nullptr) {
    return;
  }
  if (value->items != nullptr) {
    for (size_t i = 0; i < value->length; ++i) {
      spc_int_array_destroy(&value->items[i].ids);
    }
    std::free(value->items);
  }
  value->items = nullptr;
  value->length = 0;
}

void spc_scored_bytes_array_list_destroy(
    spc_scored_bytes_array_list_t *value) {
  if (value == nullptr) {
    return;
  }
  if (value->items != nullptr) {
    for (size_t i = 0; i < value->length; ++i) {
      spc_bytes_array_destroy(&value->items[i].pieces);
    }
    std::free(value->items);
  }
  value->items = nullptr;
  value->length = 0;
}

void spc_normalized_result_destroy(spc_normalized_result_t *value) {
  if (value == nullptr) {
    return;
  }
  spc_bytes_destroy(&value->normalized);
  spc_size_array_destroy(&value->offsets);
}

// -----------------------------------------------------------------------------
// Options helpers
// -----------------------------------------------------------------------------

void spc_encode_options_init(spc_encode_options_t *options) {
  if (options == nullptr) {
    return;
  }
  options->add_bos = false;
  options->add_eos = false;
  options->reverse = false;
  options->emit_unk_piece = false;
  options->enable_sampling = false;
  options->nbest_size = -1;
  options->alpha = 0.1f;
}

void spc_sample_encode_and_score_options_init(
    spc_sample_encode_and_score_options_t *options) {
  if (options == nullptr) {
    return;
  }
  options->add_bos = false;
  options->add_eos = false;
  options->reverse = false;
  options->emit_unk_piece = false;
  options->num_samples = 1;
  options->alpha = 1.0f;
  options->wor = false;
  options->include_best = false;
}

// -----------------------------------------------------------------------------
// SentencePieceProcessor façade
// -----------------------------------------------------------------------------

spc_sentencepiece_processor_t *spc_sentencepiece_processor_create(void) {
  return new (std::nothrow) spc_sentencepiece_processor();
}

void spc_sentencepiece_processor_destroy(
    spc_sentencepiece_processor_t *processor) {
  delete processor;
}

spc_status_t spc_sentencepiece_processor_load_from_file(
    spc_sentencepiece_processor_t *processor, spc_string_view_t model_path) {
  if (processor == nullptr) {
    return MakeInvalidArgument("processor is null");
  }
  Status status = processor->impl.Load(ToStringView(model_path));
  return MakeStatusFromProto(status);
}

spc_status_t spc_sentencepiece_processor_load_from_serialized_proto(
    spc_sentencepiece_processor_t *processor, const void *data,
    size_t length) {
  if (processor == nullptr || (data == nullptr && length > 0)) {
    return MakeInvalidArgument("invalid arguments");
  }
  absl::string_view serialized(static_cast<const char *>(data), length);
  Status status = processor->impl.LoadFromSerializedProto(serialized);
  return MakeStatusFromProto(status);
}

spc_status_t spc_sentencepiece_processor_set_encode_extra_options(
    spc_sentencepiece_processor_t *processor, spc_string_view_t extra_option) {
  if (processor == nullptr) {
    return MakeInvalidArgument("processor is null");
  }
  return MakeStatusFromProto(
      processor->impl.SetEncodeExtraOptions(ToStringView(extra_option)));
}

spc_status_t spc_sentencepiece_processor_set_decode_extra_options(
    spc_sentencepiece_processor_t *processor, spc_string_view_t extra_option) {
  if (processor == nullptr) {
    return MakeInvalidArgument("processor is null");
  }
  return MakeStatusFromProto(
      processor->impl.SetDecodeExtraOptions(ToStringView(extra_option)));
}

spc_status_t spc_sentencepiece_processor_set_vocabulary(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *pieces,
    size_t length) {
  if (processor == nullptr) {
    return MakeInvalidArgument("processor is null");
  }
  if (pieces == nullptr && length > 0) {
    return MakeInvalidArgument("pieces is null");
  }
  std::vector<absl::string_view> vocab;
  vocab.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    vocab.push_back(ToStringView(pieces[i]));
  }
  return MakeStatusFromProto(processor->impl.SetVocabulary(vocab));
}

spc_status_t spc_sentencepiece_processor_reset_vocabulary(
    spc_sentencepiece_processor_t *processor) {
  if (processor == nullptr) {
    return MakeInvalidArgument("processor is null");
  }
  return MakeStatusFromProto(processor->impl.ResetVocabulary());
}

spc_status_t spc_sentencepiece_processor_load_vocabulary(
    spc_sentencepiece_processor_t *processor, spc_string_view_t filename,
    int threshold) {
  if (processor == nullptr) {
    return MakeInvalidArgument("processor is null");
  }
  return MakeStatusFromProto(
      processor->impl.LoadVocabulary(ToStringView(filename), threshold));
}

int spc_sentencepiece_processor_get_piece_size(
    const spc_sentencepiece_processor_t *processor) {
  if (processor == nullptr) {
    return 0;
  }
  return processor->impl.GetPieceSize();
}

int spc_sentencepiece_processor_piece_to_id(
    const spc_sentencepiece_processor_t *processor, spc_string_view_t piece) {
  if (processor == nullptr) {
    return -1;
  }
  return processor->impl.PieceToId(ToStringView(piece));
}

spc_status_t spc_sentencepiece_processor_id_to_piece(
    const spc_sentencepiece_processor_t *processor, int id,
    spc_bytes_t *out_piece) {
  if (processor == nullptr) {
    return MakeInvalidArgument("processor is null");
  }
  if (out_piece == nullptr) {
    return MakeInvalidArgument("out_piece is null");
  }
  if (IsIdOutOfRange(id, processor->impl.GetPieceSize())) {
    return MakeStatus(SPC_STATUS_OUT_OF_RANGE, "piece id is out of range");
  }
  return AssignBytes(processor->impl.IdToPiece(id), out_piece);
}

float spc_sentencepiece_processor_get_score(
    const spc_sentencepiece_processor_t *processor, int id) {
  if (processor == nullptr) {
    return 0.0f;
  }
  return processor->impl.GetScore(id);
}

bool spc_sentencepiece_processor_is_unknown(
    const spc_sentencepiece_processor_t *processor, int id) {
  if (processor == nullptr) {
    return true;
  }
  return processor->impl.IsUnknown(id);
}

bool spc_sentencepiece_processor_is_control(
    const spc_sentencepiece_processor_t *processor, int id) {
  if (processor == nullptr) {
    return false;
  }
  return processor->impl.IsControl(id);
}

bool spc_sentencepiece_processor_is_unused(
    const spc_sentencepiece_processor_t *processor, int id) {
  if (processor == nullptr) {
    return false;
  }
  return processor->impl.IsUnused(id);
}

bool spc_sentencepiece_processor_is_byte(
    const spc_sentencepiece_processor_t *processor, int id) {
  if (processor == nullptr) {
    return false;
  }
  return processor->impl.IsByte(id);
}

int spc_sentencepiece_processor_unk_id(
    const spc_sentencepiece_processor_t *processor) {
  if (processor == nullptr) {
    return -1;
  }
  return processor->impl.unk_id();
}

int spc_sentencepiece_processor_bos_id(
    const spc_sentencepiece_processor_t *processor) {
  if (processor == nullptr) {
    return -1;
  }
  return processor->impl.bos_id();
}

int spc_sentencepiece_processor_eos_id(
    const spc_sentencepiece_processor_t *processor) {
  if (processor == nullptr) {
    return -1;
  }
  return processor->impl.eos_id();
}

int spc_sentencepiece_processor_pad_id(
    const spc_sentencepiece_processor_t *processor) {
  if (processor == nullptr) {
    return -1;
  }
  return processor->impl.pad_id();
}

spc_status_t spc_sentencepiece_processor_serialized_model_proto(
    const spc_sentencepiece_processor_t *processor, spc_bytes_t *out_model) {
  if (processor == nullptr || out_model == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  return AssignBytes(processor->impl.serialized_model_proto(), out_model);
}

spc_status_t spc_sentencepiece_processor_encode_ids(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_encode_options_t *options, spc_int_array_t *out_ids) {
  if (processor == nullptr || out_ids == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  spc_encode_options_t opts = GetEncodeOptionsOrDefault(options);
  std::vector<int> ids;
  Status status;
  if (opts.enable_sampling) {
    status = processor->impl.SampleEncode(ToStringView(input),
                                          opts.nbest_size, opts.alpha, &ids);
  } else {
    status = processor->impl.Encode(ToStringView(input), &ids);
  }
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  RewriteIds(processor->impl, &ids, opts);
  return AssignIntArray(ids, out_ids);
}

spc_status_t spc_sentencepiece_processor_encode_pieces(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_encode_options_t *options, spc_bytes_array_t *out_pieces) {
  if (processor == nullptr || out_pieces == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  spc_encode_options_t opts = GetEncodeOptionsOrDefault(options);
  std::vector<std::string> pieces;
  Status status;
  if (opts.enable_sampling) {
    status = processor->impl.SampleEncode(ToStringView(input),
                                          opts.nbest_size, opts.alpha,
                                          &pieces);
  } else {
    status = processor->impl.Encode(ToStringView(input), &pieces);
  }
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  RewritePieces(processor->impl, &pieces, opts);
  return AssignBytesArray(pieces, out_pieces);
}

spc_status_t spc_sentencepiece_processor_encode_serialized_proto(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_encode_options_t *options, spc_bytes_t *out_proto) {
  if (processor == nullptr || out_proto == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  spc_encode_options_t opts = GetEncodeOptionsOrDefault(options);
  if (RequiresProtoRewrite(opts)) {
    return MakeStatus(SPC_STATUS_UNIMPLEMENTED,
                      "proto output does not support sequence rewrite options");
  }
  return AssignBytes(processor->impl.EncodeAsSerializedProto(
                          ToStringView(input)),
                      out_proto);
}

spc_status_t spc_sentencepiece_processor_encode_ids_batch(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *inputs,
    size_t input_count, int32_t num_threads,
    const spc_encode_options_t *options, spc_int_array_list_t *out_ids) {
  if (processor == nullptr || out_ids == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (inputs == nullptr && input_count > 0) {
    return MakeInvalidArgument("inputs is null");
  }
  spc_encode_options_t opts = GetEncodeOptionsOrDefault(options);
  std::vector<std::vector<int>> results(input_count);
  spc_status_t status = ParallelFor(
      input_count, num_threads, [&](size_t index) -> spc_status_t {
        std::vector<int> local;
        Status encode_status;
        if (opts.enable_sampling) {
          encode_status = processor->impl.SampleEncode(
              ToStringView(inputs[index]), opts.nbest_size, opts.alpha,
              &local);
        } else {
          encode_status = processor->impl.Encode(
              ToStringView(inputs[index]), &local);
        }
        if (!encode_status.ok()) {
          return MakeStatusFromProto(encode_status);
        }
        RewriteIds(processor->impl, &local, opts);
        results[index] = std::move(local);
        return spc_status_t{SPC_STATUS_OK, nullptr};
      });
  if (!spc_status_is_ok(&status)) {
    return status;
  }
  return AssignIntArrayList(results, out_ids);
}

spc_status_t spc_sentencepiece_processor_encode_pieces_batch(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *inputs,
    size_t input_count, int32_t num_threads,
    const spc_encode_options_t *options, spc_bytes_array_list_t *out_pieces) {
  if (processor == nullptr || out_pieces == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (inputs == nullptr && input_count > 0) {
    return MakeInvalidArgument("inputs is null");
  }
  spc_encode_options_t opts = GetEncodeOptionsOrDefault(options);
  std::vector<std::vector<std::string>> results(input_count);
  spc_status_t status = ParallelFor(
      input_count, num_threads, [&](size_t index) -> spc_status_t {
        std::vector<std::string> local;
        Status encode_status;
        if (opts.enable_sampling) {
          encode_status = processor->impl.SampleEncode(
              ToStringView(inputs[index]), opts.nbest_size, opts.alpha,
              &local);
        } else {
          encode_status = processor->impl.Encode(
              ToStringView(inputs[index]), &local);
        }
        if (!encode_status.ok()) {
          return MakeStatusFromProto(encode_status);
        }
        RewritePieces(processor->impl, &local, opts);
        results[index] = std::move(local);
        return spc_status_t{SPC_STATUS_OK, nullptr};
      });
  if (!spc_status_is_ok(&status)) {
    return status;
  }
  return AssignBytesArrayList(results, out_pieces);
}

spc_status_t spc_sentencepiece_processor_encode_serialized_proto_batch(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *inputs,
    size_t input_count, int32_t num_threads,
    const spc_encode_options_t *options, spc_bytes_array_t *out_proto_list) {
  if (processor == nullptr || out_proto_list == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (RequiresProtoRewrite(GetEncodeOptionsOrDefault(options))) {
    return MakeStatus(SPC_STATUS_UNIMPLEMENTED,
                      "proto output does not support sequence rewrite options");
  }
  if (inputs == nullptr && input_count > 0) {
    return MakeInvalidArgument("inputs is null");
  }
  std::vector<std::string> results(input_count);
  spc_status_t status = ParallelFor(
      input_count, num_threads, [&](size_t index) -> spc_status_t {
        results[index] = processor->impl.EncodeAsSerializedProto(
            ToStringView(inputs[index]));
        return spc_status_t{SPC_STATUS_OK, nullptr};
      });
  if (!spc_status_is_ok(&status)) {
    return status;
  }
  return AssignBytesArrayFlat(results, out_proto_list);
}

spc_status_t spc_sentencepiece_processor_decode_ids(
    spc_sentencepiece_processor_t *processor, const int32_t *ids,
    size_t length, spc_bytes_t *out_text) {
  if (processor == nullptr || out_text == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (ids == nullptr && length > 0) {
    return MakeInvalidArgument("ids is null");
  }
  std::vector<int> values(ids, ids + length);
  for (int value : values) {
    if (IsIdOutOfRange(value, processor->impl.GetPieceSize())) {
      return MakeStatus(SPC_STATUS_OUT_OF_RANGE,
                        "piece id is out of range");
    }
  }
  std::string text;
  Status status = processor->impl.Decode(values, &text);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  return AssignBytes(text, out_text);
}

spc_status_t spc_sentencepiece_processor_decode_ids_as_bytes(
    spc_sentencepiece_processor_t *processor, const int32_t *ids,
    size_t length, spc_bytes_t *out_bytes) {
  return spc_sentencepiece_processor_decode_ids(processor, ids, length,
                                                out_bytes);
}

spc_status_t spc_sentencepiece_processor_decode_pieces(
    spc_sentencepiece_processor_t *processor,
    const spc_string_view_t *pieces, size_t length,
    spc_bytes_t *out_text) {
  if (processor == nullptr || out_text == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (pieces == nullptr && length > 0) {
    return MakeInvalidArgument("pieces is null");
  }
  std::vector<absl::string_view> seq;
  seq.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    seq.push_back(ToStringView(pieces[i]));
  }
  std::string text;
  Status status = processor->impl.Decode(seq, &text);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  return AssignBytes(text, out_text);
}

spc_status_t spc_sentencepiece_processor_decode_ids_as_serialized_proto(
    spc_sentencepiece_processor_t *processor, const int32_t *ids,
    size_t length, spc_bytes_t *out_proto) {
  if (processor == nullptr || out_proto == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (ids == nullptr && length > 0) {
    return MakeInvalidArgument("ids is null");
  }
  std::vector<int> values(ids, ids + length);
  for (int value : values) {
    if (IsIdOutOfRange(value, processor->impl.GetPieceSize())) {
      return MakeStatus(SPC_STATUS_OUT_OF_RANGE,
                        "piece id is out of range");
    }
  }
  return AssignBytes(
      processor->impl.DecodeIdsAsSerializedProto(values), out_proto);
}

spc_status_t spc_sentencepiece_processor_decode_pieces_as_serialized_proto(
    spc_sentencepiece_processor_t *processor,
    const spc_string_view_t *pieces, size_t length, spc_bytes_t *out_proto) {
  if (processor == nullptr || out_proto == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (pieces == nullptr && length > 0) {
    return MakeInvalidArgument("pieces is null");
  }
  std::vector<absl::string_view> seq;
  seq.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    seq.push_back(ToStringView(pieces[i]));
  }
  return AssignBytes(
      processor->impl.DecodePiecesAsSerializedProto(seq), out_proto);
}

spc_status_t spc_sentencepiece_processor_decode_ids_batch(
    spc_sentencepiece_processor_t *processor, const spc_int_span_t *inputs,
    size_t input_count, int32_t num_threads,
    spc_bytes_array_t *out_texts) {
  if (processor == nullptr || out_texts == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (inputs == nullptr && input_count > 0) {
    return MakeInvalidArgument("inputs is null");
  }
  std::vector<std::string> results(input_count);
  spc_status_t status = ParallelFor(
      input_count, num_threads, [&](size_t index) -> spc_status_t {
        const spc_int_span_t &span = inputs[index];
        std::vector<int> values(span.data, span.data + span.length);
        for (int value : values) {
          if (IsIdOutOfRange(value, processor->impl.GetPieceSize())) {
            return MakeStatus(SPC_STATUS_OUT_OF_RANGE,
                              "piece id is out of range");
          }
        }
        std::string text;
        Status decode_status = processor->impl.Decode(values, &text);
        if (!decode_status.ok()) {
          return MakeStatusFromProto(decode_status);
        }
        results[index] = std::move(text);
        return spc_status_t{SPC_STATUS_OK, nullptr};
      });
  if (!spc_status_is_ok(&status)) {
    return status;
  }
  return AssignBytesArrayFlat(results, out_texts);
}

spc_status_t spc_sentencepiece_processor_decode_ids_as_bytes_batch(
    spc_sentencepiece_processor_t *processor, const spc_int_span_t *inputs,
    size_t input_count, int32_t num_threads,
    spc_bytes_array_t *out_texts) {
  return spc_sentencepiece_processor_decode_ids_batch(
      processor, inputs, input_count, num_threads, out_texts);
}

spc_status_t spc_sentencepiece_processor_decode_ids_as_serialized_proto_batch(
    spc_sentencepiece_processor_t *processor, const spc_int_span_t *inputs,
    size_t input_count, int32_t num_threads,
    spc_bytes_array_t *out_proto_list) {
  if (processor == nullptr || out_proto_list == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (inputs == nullptr && input_count > 0) {
    return MakeInvalidArgument("inputs is null");
  }
  std::vector<std::string> results(input_count);
  spc_status_t status = ParallelFor(
      input_count, num_threads, [&](size_t index) -> spc_status_t {
        const spc_int_span_t &span = inputs[index];
        std::vector<int> values(span.data, span.data + span.length);
        for (int value : values) {
          if (IsIdOutOfRange(value, processor->impl.GetPieceSize())) {
            return MakeStatus(SPC_STATUS_OUT_OF_RANGE,
                              "piece id is out of range");
          }
        }
        results[index] =
            processor->impl.DecodeIdsAsSerializedProto(values);
        return spc_status_t{SPC_STATUS_OK, nullptr};
      });
  if (!spc_status_is_ok(&status)) {
    return status;
  }
  return AssignBytesArrayFlat(results, out_proto_list);
}

spc_status_t spc_sentencepiece_processor_decode_pieces_batch(
    spc_sentencepiece_processor_t *processor,
    const spc_string_view_span_t *inputs, size_t input_count,
    int32_t num_threads, spc_bytes_array_t *out_texts) {
  if (processor == nullptr || out_texts == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (inputs == nullptr && input_count > 0) {
    return MakeInvalidArgument("inputs is null");
  }
  std::vector<std::string> results(input_count);
  spc_status_t status = ParallelFor(
      input_count, num_threads, [&](size_t index) -> spc_status_t {
        const spc_string_view_span_t &span = inputs[index];
        std::vector<absl::string_view> pieces;
        pieces.reserve(span.length);
        for (size_t i = 0; i < span.length; ++i) {
          pieces.push_back(ToStringView(span.items[i]));
        }
        std::string text;
        Status decode_status = processor->impl.Decode(pieces, &text);
        if (!decode_status.ok()) {
          return MakeStatusFromProto(decode_status);
        }
        results[index] = std::move(text);
        return spc_status_t{SPC_STATUS_OK, nullptr};
      });
  if (!spc_status_is_ok(&status)) {
    return status;
  }
  return AssignBytesArrayFlat(results, out_texts);
}

spc_status_t
spc_sentencepiece_processor_decode_pieces_as_serialized_proto_batch(
    spc_sentencepiece_processor_t *processor,
    const spc_string_view_span_t *inputs, size_t input_count,
    int32_t num_threads, spc_bytes_array_t *out_proto_list) {
  if (processor == nullptr || out_proto_list == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (inputs == nullptr && input_count > 0) {
    return MakeInvalidArgument("inputs is null");
  }
  std::vector<std::string> results(input_count);
  spc_status_t status = ParallelFor(
      input_count, num_threads, [&](size_t index) -> spc_status_t {
        const spc_string_view_span_t &span = inputs[index];
        std::vector<absl::string_view> pieces;
        pieces.reserve(span.length);
        for (size_t i = 0; i < span.length; ++i) {
          pieces.push_back(ToStringView(span.items[i]));
        }
        results[index] =
            processor->impl.DecodePiecesAsSerializedProto(pieces);
        return spc_status_t{SPC_STATUS_OK, nullptr};
      });
  if (!spc_status_is_ok(&status)) {
    return status;
  }
  return AssignBytesArrayFlat(results, out_proto_list);
}

spc_status_t spc_sentencepiece_processor_nbest_encode_ids(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    int nbest_size, const spc_encode_options_t *options,
    spc_int_array_list_t *out_lists) {
  if (processor == nullptr || out_lists == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  spc_encode_options_t opts = GetEncodeOptionsOrDefault(options);
  std::vector<std::vector<int>> results;
  Status status = processor->impl.NBestEncode(ToStringView(input), nbest_size,
                                              &results);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  for (auto &ids : results) {
    RewriteIds(processor->impl, &ids, opts);
  }
  return AssignIntArrayList(results, out_lists);
}

spc_status_t spc_sentencepiece_processor_nbest_encode_pieces(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    int nbest_size, const spc_encode_options_t *options,
    spc_bytes_array_list_t *out_lists) {
  if (processor == nullptr || out_lists == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  spc_encode_options_t opts = GetEncodeOptionsOrDefault(options);
  std::vector<std::vector<std::string>> results;
  Status status = processor->impl.NBestEncode(ToStringView(input), nbest_size,
                                              &results);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  for (auto &pieces : results) {
    RewritePieces(processor->impl, &pieces, opts);
  }
  return AssignBytesArrayList(results, out_lists);
}

spc_status_t spc_sentencepiece_processor_nbest_encode_serialized_proto(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    int nbest_size, const spc_encode_options_t *options,
    spc_bytes_t *out_proto) {
  if (processor == nullptr || out_proto == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (RequiresProtoRewrite(GetEncodeOptionsOrDefault(options))) {
    return MakeStatus(SPC_STATUS_UNIMPLEMENTED,
                      "proto output does not support sequence rewrite options");
  }
  return AssignBytes(processor->impl.NBestEncodeAsSerializedProto(
                          ToStringView(input), nbest_size),
                      out_proto);
}

spc_status_t spc_sentencepiece_processor_sample_encode_and_score_ids(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_sample_encode_and_score_options_t *options,
    spc_scored_int_array_list_t *out_lists) {
  if (processor == nullptr || out_lists == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  spc_sample_encode_and_score_options_t opts =
      GetSampleOptionsOrDefault(options);
  std::vector<std::pair<std::vector<int>, float>> results;
  Status status = processor->impl.SampleEncodeAndScore(
      ToStringView(input), opts.num_samples, opts.alpha, opts.wor,
      opts.include_best, &results);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  for (auto &entry : results) {
    RewriteIds(processor->impl, &entry.first, FromSampleOptions(opts));
  }
  return AssignScoredIntArrayList(results, out_lists);
}

spc_status_t spc_sentencepiece_processor_sample_encode_and_score_pieces(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_sample_encode_and_score_options_t *options,
    spc_scored_bytes_array_list_t *out_lists) {
  if (processor == nullptr || out_lists == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  spc_sample_encode_and_score_options_t opts =
      GetSampleOptionsOrDefault(options);
  std::vector<std::pair<std::vector<std::string>, float>> results;
  Status status = processor->impl.SampleEncodeAndScore(
      ToStringView(input), opts.num_samples, opts.alpha, opts.wor,
      opts.include_best, &results);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  for (auto &entry : results) {
    RewritePieces(processor->impl, &entry.first, FromSampleOptions(opts));
  }
  return AssignScoredBytesArrayList(results, out_lists);
}

spc_status_t
spc_sentencepiece_processor_sample_encode_and_score_serialized_proto(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_sample_encode_and_score_options_t *options,
    spc_bytes_t *out_proto) {
  if (processor == nullptr || out_proto == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  spc_sample_encode_and_score_options_t opts =
      GetSampleOptionsOrDefault(options);
  if (RequiresProtoRewrite(FromSampleOptions(opts))) {
    return MakeStatus(SPC_STATUS_UNIMPLEMENTED,
                      "proto output does not support sequence rewrite options");
  }
  return AssignBytes(
      processor->impl.SampleEncodeAndScoreAsSerializedProto(
          ToStringView(input), opts.num_samples, opts.alpha, opts.wor,
          opts.include_best),
      out_proto);
}

spc_status_t spc_sentencepiece_processor_normalize(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    spc_bytes_t *out_text) {
  if (processor == nullptr || out_text == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  std::string normalized;
  Status status = processor->impl.Normalize(ToStringView(input), &normalized);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  return AssignBytes(normalized, out_text);
}

spc_status_t spc_sentencepiece_processor_normalize_with_offsets(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    spc_normalized_result_t *out_result) {
  if (processor == nullptr || out_result == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  std::string normalized;
  std::vector<size_t> offsets;
  Status status = processor->impl.Normalize(ToStringView(input), &normalized,
                                            &offsets);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  spc_status_t assign_status = AssignBytes(normalized, &out_result->normalized);
  if (!spc_status_is_ok(&assign_status)) {
    return assign_status;
  }
  out_result->offsets.data = nullptr;
  out_result->offsets.length = 0;
  if (!offsets.empty()) {
    size_t *buffer = static_cast<size_t *>(
        std::malloc(sizeof(size_t) * offsets.size()));
    if (buffer == nullptr) {
      spc_bytes_destroy(&out_result->normalized);
      return MakeResourceExhausted("memory allocation failed");
    }
    std::memcpy(buffer, offsets.data(), sizeof(size_t) * offsets.size());
    out_result->offsets.data = buffer;
    out_result->offsets.length = offsets.size();
  }
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t spc_sentencepiece_processor_calculate_entropy(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    float alpha, float *out_entropy) {
  if (processor == nullptr || out_entropy == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  float entropy = 0.0f;
  Status status = processor->impl.CalculateEntropy(ToStringView(input), alpha,
                                                   &entropy);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  *out_entropy = entropy;
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t spc_sentencepiece_processor_calculate_entropy_batch(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *inputs,
    size_t input_count, float alpha, int32_t num_threads,
    spc_float_array_t *out_entropies) {
  if (processor == nullptr || out_entropies == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  if (inputs == nullptr && input_count > 0) {
    return MakeInvalidArgument("inputs is null");
  }
  std::vector<float> entropies(input_count, 0.0f);
  spc_status_t status = ParallelFor(
      input_count, num_threads, [&](size_t index) -> spc_status_t {
        float value = 0.0f;
        Status entropy_status = processor->impl.CalculateEntropy(
            ToStringView(inputs[index]), alpha, &value);
        if (!entropy_status.ok()) {
          return MakeStatusFromProto(entropy_status);
        }
        entropies[index] = value;
        return spc_status_t{SPC_STATUS_OK, nullptr};
      });
  if (!spc_status_is_ok(&status)) {
    return status;
  }
  return AssignFloatArray(entropies, out_entropies);
}

spc_status_t spc_sentencepiece_processor_override_normalizer_spec(
    spc_sentencepiece_processor_t *processor, const spc_map_entry_t *entries,
    size_t length) {
  if (processor == nullptr) {
    return MakeInvalidArgument("processor is null");
  }
  if (entries == nullptr && length > 0) {
    return MakeInvalidArgument("entries is null");
  }
  for (size_t i = 0; i < length; ++i) {
    const auto &entry = entries[i];
    Status status = SentencePieceTrainer::SetProtoField(
        ToStringView(entry.key), ToStringView(entry.value),
        processor->impl.mutable_normalizer_spec());
    if (!status.ok()) {
      return MakeStatusFromProto(status);
    }
  }
  return {SPC_STATUS_OK, nullptr};
}

// -----------------------------------------------------------------------------
// SentencePieceTrainer façade
// -----------------------------------------------------------------------------

spc_status_t spc_sentencepiece_trainer_train_from_string(
    spc_string_view_t args, spc_bytes_t *out_model) {
  std::string serialized;
  absl::string_view args_view = ToStringView(args);
  std::string adjusted_args;
  if (out_model != nullptr) {
    adjusted_args.assign(args_view.begin(), args_view.end());
    if (!adjusted_args.empty()) {
      adjusted_args.append(" ");
    }
    adjusted_args.append("--model_prefix=");
    args_view = absl::string_view(adjusted_args);
  }
  Status status = SentencePieceTrainer::Train(args_view, nullptr,
                                              out_model ? &serialized : nullptr);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  if (out_model != nullptr) {
    return AssignBytes(serialized, out_model);
  }
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t spc_sentencepiece_trainer_train_from_string_with_sentences(
    spc_string_view_t args, const spc_string_view_t *sentences,
    size_t sentence_count, spc_bytes_t *out_model) {
  if (sentences == nullptr && sentence_count > 0) {
    return MakeInvalidArgument("sentences is null");
  }
  std::vector<std::string> corpus;
  corpus.reserve(sentence_count);
  for (size_t i = 0; i < sentence_count; ++i) {
    corpus.emplace_back(sentences[i].data, sentences[i].length);
  }
  std::string serialized;
  absl::string_view args_view = ToStringView(args);
  std::string adjusted_args;
  if (out_model != nullptr) {
    adjusted_args.assign(args_view.begin(), args_view.end());
    if (!adjusted_args.empty()) {
      adjusted_args.append(" ");
    }
    adjusted_args.append("--model_prefix=");
    args_view = absl::string_view(adjusted_args);
  }
  Status status = SentencePieceTrainer::Train(args_view, corpus,
                                              out_model ? &serialized : nullptr);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  if (out_model != nullptr) {
    return AssignBytes(serialized, out_model);
  }
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t spc_sentencepiece_trainer_train_from_map(
    const spc_map_entry_t *entries, size_t entry_count,
    spc_bytes_t *out_model) {
  if (entries == nullptr && entry_count > 0) {
    return MakeInvalidArgument("entries is null");
  }
  std::unordered_map<std::string, std::string> kwargs;
  kwargs.reserve(entry_count);
  for (size_t i = 0; i < entry_count; ++i) {
    kwargs.emplace(std::string(entries[i].key.data, entries[i].key.length),
                   std::string(entries[i].value.data, entries[i].value.length));
  }
  if (out_model != nullptr) {
    kwargs["model_prefix"] = "";
  }
  std::string serialized;
  Status status = SentencePieceTrainer::Train(kwargs, nullptr,
                                              out_model ? &serialized : nullptr);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  if (out_model != nullptr) {
    return AssignBytes(serialized, out_model);
  }
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t spc_sentencepiece_trainer_train_from_map_with_sentences(
    const spc_map_entry_t *entries, size_t entry_count,
    const spc_string_view_t *sentences, size_t sentence_count,
    spc_bytes_t *out_model) {
  if ((entries == nullptr && entry_count > 0) ||
      (sentences == nullptr && sentence_count > 0)) {
    return MakeInvalidArgument("invalid arguments");
  }
  std::unordered_map<std::string, std::string> kwargs;
  kwargs.reserve(entry_count);
  for (size_t i = 0; i < entry_count; ++i) {
    kwargs.emplace(std::string(entries[i].key.data, entries[i].key.length),
                   std::string(entries[i].value.data, entries[i].value.length));
  }
  if (out_model != nullptr) {
    kwargs["model_prefix"] = "";
  }
  std::vector<std::string> corpus;
  corpus.reserve(sentence_count);
  for (size_t i = 0; i < sentence_count; ++i) {
    corpus.emplace_back(sentences[i].data, sentences[i].length);
  }
  std::string serialized;
  Status status = SentencePieceTrainer::Train(kwargs, corpus,
                                              out_model ? &serialized : nullptr);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  if (out_model != nullptr) {
    return AssignBytes(serialized, out_model);
  }
  return {SPC_STATUS_OK, nullptr};
}

// -----------------------------------------------------------------------------
// SentencePieceNormalizer façade
// -----------------------------------------------------------------------------

spc_sentencepiece_normalizer_t *spc_sentencepiece_normalizer_create(void) {
  return new (std::nothrow) spc_sentencepiece_normalizer();
}

void spc_sentencepiece_normalizer_destroy(
    spc_sentencepiece_normalizer_t *normalizer) {
  delete normalizer;
}

spc_status_t spc_sentencepiece_normalizer_load_from_file(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t filename) {
  if (normalizer == nullptr) {
    return MakeInvalidArgument("normalizer is null");
  }
  return MakeStatusFromProto(
      normalizer->impl.Load(ToStringView(filename)));
}

spc_status_t spc_sentencepiece_normalizer_load_from_serialized_proto(
    spc_sentencepiece_normalizer_t *normalizer, const void *data,
    size_t length) {
  if (normalizer == nullptr || (data == nullptr && length > 0)) {
    return MakeInvalidArgument("invalid arguments");
  }
  absl::string_view serialized(static_cast<const char *>(data), length);
  return MakeStatusFromProto(normalizer->impl.LoadFromSerializedProto(
      serialized));
}

spc_status_t spc_sentencepiece_normalizer_load_from_rule_tsv(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t filename) {
  if (normalizer == nullptr) {
    return MakeInvalidArgument("normalizer is null");
  }
  return MakeStatusFromProto(
      normalizer->impl.LoadFromRuleTSV(ToStringView(filename)));
}

spc_status_t spc_sentencepiece_normalizer_load_from_rule_name(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t name) {
  if (normalizer == nullptr) {
    return MakeInvalidArgument("normalizer is null");
  }
  return MakeStatusFromProto(
      normalizer->impl.LoadFromRuleName(ToStringView(name)));
}

spc_status_t spc_sentencepiece_normalizer_normalize(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t input,
    spc_bytes_t *out_text) {
  if (normalizer == nullptr || out_text == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  std::string normalized;
  Status status = normalizer->impl.Normalize(ToStringView(input), &normalized);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  return AssignBytes(normalized, out_text);
}

spc_status_t spc_sentencepiece_normalizer_normalize_with_offsets(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t input,
    spc_normalized_result_t *out_result) {
  if (normalizer == nullptr || out_result == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  std::string normalized;
  std::vector<size_t> offsets;
  Status status = normalizer->impl.Normalize(ToStringView(input), &normalized,
                                             &offsets);
  if (!status.ok()) {
    return MakeStatusFromProto(status);
  }
  spc_status_t assign_status = AssignBytes(normalized, &out_result->normalized);
  if (!spc_status_is_ok(&assign_status)) {
    return assign_status;
  }
  out_result->offsets.data = nullptr;
  out_result->offsets.length = 0;
  if (!offsets.empty()) {
    size_t *buffer = static_cast<size_t *>(
        std::malloc(sizeof(size_t) * offsets.size()));
    if (buffer == nullptr) {
      spc_bytes_destroy(&out_result->normalized);
      return MakeResourceExhausted("memory allocation failed");
    }
    std::memcpy(buffer, offsets.data(), sizeof(size_t) * offsets.size());
    out_result->offsets.data = buffer;
    out_result->offsets.length = offsets.size();
  }
  return {SPC_STATUS_OK, nullptr};
}

spc_status_t spc_sentencepiece_normalizer_serialized_model_proto(
    spc_sentencepiece_normalizer_t *normalizer, spc_bytes_t *out_model) {
  if (normalizer == nullptr || out_model == nullptr) {
    return MakeInvalidArgument("invalid arguments");
  }
  return AssignBytes(normalizer->impl.serialized_model_proto(), out_model);
}

// -----------------------------------------------------------------------------
// Global utilities
// -----------------------------------------------------------------------------

void spc_set_random_generator_seed(uint32_t seed) {
  sentencepiece::SetRandomGeneratorSeed(seed);
}

void spc_set_min_log_level(int level) {
  sentencepiece::SetMinLogLevel(level);
}

void spc_set_data_dir(spc_string_view_t data_dir) {
  sentencepiece::SetDataDir(ToStringView(data_dir));
}

}  // extern "C"
