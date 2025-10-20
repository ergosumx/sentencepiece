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

#ifndef SENTENCEPIECE_BINDINGS_C_SENTENCEPIECE_C_H_
#define SENTENCEPIECE_BINDINGS_C_SENTENCEPIECE_C_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && !defined(SPC_STATIC)
#if defined(SPC_EXPORTS)
#define SPC_API __declspec(dllexport)
#else
#define SPC_API __declspec(dllimport)
#endif
#else
#define SPC_API
#endif

// -----------------------------------------------------------------------------
// Status helpers
// -----------------------------------------------------------------------------

typedef enum spc_status_code {
  SPC_STATUS_OK = 0,
  SPC_STATUS_CANCELLED = 1,
  SPC_STATUS_UNKNOWN = 2,
  SPC_STATUS_INVALID_ARGUMENT = 3,
  SPC_STATUS_DEADLINE_EXCEEDED = 4,
  SPC_STATUS_NOT_FOUND = 5,
  SPC_STATUS_ALREADY_EXISTS = 6,
  SPC_STATUS_PERMISSION_DENIED = 7,
  SPC_STATUS_RESOURCE_EXHAUSTED = 8,
  SPC_STATUS_FAILED_PRECONDITION = 9,
  SPC_STATUS_ABORTED = 10,
  SPC_STATUS_OUT_OF_RANGE = 11,
  SPC_STATUS_UNIMPLEMENTED = 12,
  SPC_STATUS_INTERNAL = 13,
  SPC_STATUS_UNAVAILABLE = 14,
  SPC_STATUS_DATA_LOSS = 15,
  SPC_STATUS_UNAUTHENTICATED = 16
} spc_status_code_t;

typedef struct spc_status {
  spc_status_code_t code;
  char *message;
} spc_status_t;

SPC_API void spc_status_destroy(spc_status_t *status);
SPC_API bool spc_status_is_ok(const spc_status_t *status);

// -----------------------------------------------------------------------------
// Common value containers
// -----------------------------------------------------------------------------

typedef struct spc_string_view {
  const char *data;
  size_t length;
} spc_string_view_t;

typedef struct spc_string_view_span {
  const spc_string_view_t *items;
  size_t length;
} spc_string_view_span_t;

SPC_API spc_string_view_t spc_string_view_from_cstr(const char *value);

typedef struct spc_int_span {
  const int32_t *data;
  size_t length;
} spc_int_span_t;


typedef struct spc_bytes {
  char *data;
  size_t length;
} spc_bytes_t;

SPC_API void spc_bytes_destroy(spc_bytes_t *value);


typedef struct spc_int_array {
  int32_t *data;
  size_t length;
} spc_int_array_t;

SPC_API void spc_int_array_destroy(spc_int_array_t *value);


typedef struct spc_float_array {
  float *data;
  size_t length;
} spc_float_array_t;

SPC_API void spc_float_array_destroy(spc_float_array_t *value);


typedef struct spc_size_array {
  size_t *data;
  size_t length;
} spc_size_array_t;

SPC_API void spc_size_array_destroy(spc_size_array_t *value);


typedef struct spc_int_array_list {
  spc_int_array_t *items;
  size_t length;
} spc_int_array_list_t;

SPC_API void spc_int_array_list_destroy(spc_int_array_list_t *value);


typedef struct spc_bytes_array {
  spc_bytes_t *items;
  size_t length;
} spc_bytes_array_t;

SPC_API void spc_bytes_array_destroy(spc_bytes_array_t *value);


typedef struct spc_bytes_array_list {
  spc_bytes_array_t *items;
  size_t length;
} spc_bytes_array_list_t;

SPC_API void spc_bytes_array_list_destroy(spc_bytes_array_list_t *value);


typedef struct spc_scored_int_array {
  spc_int_array_t ids;
  float score;
} spc_scored_int_array_t;


typedef struct spc_scored_int_array_list {
  spc_scored_int_array_t *items;
  size_t length;
} spc_scored_int_array_list_t;

SPC_API void spc_scored_int_array_list_destroy(spc_scored_int_array_list_t *value);


typedef struct spc_scored_bytes_array {
  spc_bytes_array_t pieces;
  float score;
} spc_scored_bytes_array_t;


typedef struct spc_scored_bytes_array_list {
  spc_scored_bytes_array_t *items;
  size_t length;
} spc_scored_bytes_array_list_t;

SPC_API void spc_scored_bytes_array_list_destroy(
    spc_scored_bytes_array_list_t *value);


typedef struct spc_normalized_result {
  spc_bytes_t normalized;
  spc_size_array_t offsets;
} spc_normalized_result_t;

SPC_API void spc_normalized_result_destroy(spc_normalized_result_t *value);


typedef struct spc_map_entry {
  spc_string_view_t key;
  spc_string_view_t value;
} spc_map_entry_t;

// -----------------------------------------------------------------------------
// Encode/Sample options
// -----------------------------------------------------------------------------

typedef struct spc_encode_options {
  bool add_bos;
  bool add_eos;
  bool reverse;
  bool emit_unk_piece;
  bool enable_sampling;
  int nbest_size;
  float alpha;
} spc_encode_options_t;

SPC_API void spc_encode_options_init(spc_encode_options_t *options);

typedef struct spc_sample_encode_and_score_options {
  bool add_bos;
  bool add_eos;
  bool reverse;
  bool emit_unk_piece;
  int num_samples;
  float alpha;
  bool wor;
  bool include_best;
} spc_sample_encode_and_score_options_t;

SPC_API void spc_sample_encode_and_score_options_init(
    spc_sample_encode_and_score_options_t *options);

// -----------------------------------------------------------------------------
// SentencePieceProcessor façade
// -----------------------------------------------------------------------------

typedef struct spc_sentencepiece_processor spc_sentencepiece_processor_t;

SPC_API spc_sentencepiece_processor_t *
spc_sentencepiece_processor_create(void);

SPC_API void spc_sentencepiece_processor_destroy(
    spc_sentencepiece_processor_t *processor);

SPC_API spc_status_t spc_sentencepiece_processor_load_from_file(
    spc_sentencepiece_processor_t *processor, spc_string_view_t model_path);

SPC_API spc_status_t spc_sentencepiece_processor_load_from_serialized_proto(
    spc_sentencepiece_processor_t *processor, const void *data, size_t length);

SPC_API spc_status_t spc_sentencepiece_processor_set_encode_extra_options(
    spc_sentencepiece_processor_t *processor, spc_string_view_t extra_option);

SPC_API spc_status_t spc_sentencepiece_processor_set_decode_extra_options(
    spc_sentencepiece_processor_t *processor, spc_string_view_t extra_option);

SPC_API spc_status_t spc_sentencepiece_processor_set_vocabulary(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *pieces,
    size_t length);

SPC_API spc_status_t spc_sentencepiece_processor_reset_vocabulary(
    spc_sentencepiece_processor_t *processor);

SPC_API spc_status_t spc_sentencepiece_processor_load_vocabulary(
    spc_sentencepiece_processor_t *processor, spc_string_view_t filename,
    int threshold);

SPC_API int spc_sentencepiece_processor_get_piece_size(
    const spc_sentencepiece_processor_t *processor);

SPC_API int spc_sentencepiece_processor_piece_to_id(
    const spc_sentencepiece_processor_t *processor, spc_string_view_t piece);

SPC_API spc_status_t spc_sentencepiece_processor_id_to_piece(
    const spc_sentencepiece_processor_t *processor, int id,
    spc_bytes_t *out_piece);

SPC_API float spc_sentencepiece_processor_get_score(
    const spc_sentencepiece_processor_t *processor, int id);

SPC_API bool spc_sentencepiece_processor_is_unknown(
    const spc_sentencepiece_processor_t *processor, int id);

SPC_API bool spc_sentencepiece_processor_is_control(
    const spc_sentencepiece_processor_t *processor, int id);

SPC_API bool spc_sentencepiece_processor_is_unused(
    const spc_sentencepiece_processor_t *processor, int id);

SPC_API bool spc_sentencepiece_processor_is_byte(
    const spc_sentencepiece_processor_t *processor, int id);

SPC_API int spc_sentencepiece_processor_unk_id(
    const spc_sentencepiece_processor_t *processor);

SPC_API int spc_sentencepiece_processor_bos_id(
    const spc_sentencepiece_processor_t *processor);

SPC_API int spc_sentencepiece_processor_eos_id(
    const spc_sentencepiece_processor_t *processor);

SPC_API int spc_sentencepiece_processor_pad_id(
    const spc_sentencepiece_processor_t *processor);

SPC_API spc_status_t spc_sentencepiece_processor_serialized_model_proto(
    const spc_sentencepiece_processor_t *processor, spc_bytes_t *out_model);

SPC_API spc_status_t spc_sentencepiece_processor_encode_ids(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_encode_options_t *options, spc_int_array_t *out_ids);

SPC_API spc_status_t spc_sentencepiece_processor_encode_pieces(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_encode_options_t *options, spc_bytes_array_t *out_pieces);

SPC_API spc_status_t spc_sentencepiece_processor_encode_serialized_proto(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_encode_options_t *options, spc_bytes_t *out_proto);

SPC_API spc_status_t spc_sentencepiece_processor_encode_ids_batch(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *inputs,
    size_t input_count, int32_t num_threads,
    const spc_encode_options_t *options, spc_int_array_list_t *out_ids);

SPC_API spc_status_t spc_sentencepiece_processor_encode_pieces_batch(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *inputs,
    size_t input_count, int32_t num_threads,
    const spc_encode_options_t *options, spc_bytes_array_list_t *out_pieces);

SPC_API spc_status_t spc_sentencepiece_processor_encode_serialized_proto_batch(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *inputs,
    size_t input_count, int32_t num_threads,
    const spc_encode_options_t *options, spc_bytes_array_t *out_proto_list);

SPC_API spc_status_t spc_sentencepiece_processor_decode_ids(
    spc_sentencepiece_processor_t *processor, const int32_t *ids,
    size_t length, spc_bytes_t *out_text);

SPC_API spc_status_t spc_sentencepiece_processor_decode_ids_as_bytes(
    spc_sentencepiece_processor_t *processor, const int32_t *ids,
    size_t length, spc_bytes_t *out_bytes);

SPC_API spc_status_t spc_sentencepiece_processor_decode_pieces(
    spc_sentencepiece_processor_t *processor,
    const spc_string_view_t *pieces, size_t length,
    spc_bytes_t *out_text);

SPC_API spc_status_t spc_sentencepiece_processor_decode_ids_as_serialized_proto(
    spc_sentencepiece_processor_t *processor, const int32_t *ids,
    size_t length, spc_bytes_t *out_proto);

SPC_API spc_status_t spc_sentencepiece_processor_decode_pieces_as_serialized_proto(
    spc_sentencepiece_processor_t *processor,
    const spc_string_view_t *pieces, size_t length, spc_bytes_t *out_proto);

SPC_API spc_status_t spc_sentencepiece_processor_decode_ids_batch(
    spc_sentencepiece_processor_t *processor, const spc_int_span_t *inputs,
    size_t input_count, int32_t num_threads,
    spc_bytes_array_t *out_texts);

SPC_API spc_status_t spc_sentencepiece_processor_decode_ids_as_bytes_batch(
    spc_sentencepiece_processor_t *processor, const spc_int_span_t *inputs,
    size_t input_count, int32_t num_threads,
    spc_bytes_array_t *out_texts);

SPC_API spc_status_t spc_sentencepiece_processor_decode_ids_as_serialized_proto_batch(
    spc_sentencepiece_processor_t *processor, const spc_int_span_t *inputs,
    size_t input_count, int32_t num_threads,
    spc_bytes_array_t *out_proto_list);

SPC_API spc_status_t spc_sentencepiece_processor_decode_pieces_batch(
    spc_sentencepiece_processor_t *processor,
    const spc_string_view_span_t *inputs, size_t input_count,
    int32_t num_threads, spc_bytes_array_t *out_texts);

SPC_API spc_status_t
spc_sentencepiece_processor_decode_pieces_as_serialized_proto_batch(
    spc_sentencepiece_processor_t *processor,
    const spc_string_view_span_t *inputs, size_t input_count,
    int32_t num_threads, spc_bytes_array_t *out_proto_list);

SPC_API spc_status_t spc_sentencepiece_processor_nbest_encode_ids(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    int nbest_size, const spc_encode_options_t *options,
    spc_int_array_list_t *out_lists);

SPC_API spc_status_t spc_sentencepiece_processor_nbest_encode_pieces(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    int nbest_size, const spc_encode_options_t *options,
    spc_bytes_array_list_t *out_lists);

SPC_API spc_status_t spc_sentencepiece_processor_nbest_encode_serialized_proto(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    int nbest_size, const spc_encode_options_t *options,
    spc_bytes_t *out_proto);

SPC_API spc_status_t spc_sentencepiece_processor_sample_encode_and_score_ids(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_sample_encode_and_score_options_t *options,
    spc_scored_int_array_list_t *out_lists);

SPC_API spc_status_t spc_sentencepiece_processor_sample_encode_and_score_pieces(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_sample_encode_and_score_options_t *options,
    spc_scored_bytes_array_list_t *out_lists);

SPC_API spc_status_t
spc_sentencepiece_processor_sample_encode_and_score_serialized_proto(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    const spc_sample_encode_and_score_options_t *options,
    spc_bytes_t *out_proto);

SPC_API spc_status_t spc_sentencepiece_processor_normalize(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    spc_bytes_t *out_text);

SPC_API spc_status_t spc_sentencepiece_processor_normalize_with_offsets(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    spc_normalized_result_t *out_result);

SPC_API spc_status_t spc_sentencepiece_processor_calculate_entropy(
    spc_sentencepiece_processor_t *processor, spc_string_view_t input,
    float alpha, float *out_entropy);

SPC_API spc_status_t spc_sentencepiece_processor_calculate_entropy_batch(
    spc_sentencepiece_processor_t *processor, const spc_string_view_t *inputs,
    size_t input_count, float alpha, int32_t num_threads,
    spc_float_array_t *out_entropies);

SPC_API spc_status_t spc_sentencepiece_processor_override_normalizer_spec(
    spc_sentencepiece_processor_t *processor, const spc_map_entry_t *entries,
    size_t length);

// -----------------------------------------------------------------------------
// SentencePieceTrainer façade
// -----------------------------------------------------------------------------

SPC_API spc_status_t spc_sentencepiece_trainer_train_from_string(
    spc_string_view_t args, spc_bytes_t *out_model);

SPC_API spc_status_t spc_sentencepiece_trainer_train_from_string_with_sentences(
    spc_string_view_t args, const spc_string_view_t *sentences,
    size_t sentence_count, spc_bytes_t *out_model);

SPC_API spc_status_t spc_sentencepiece_trainer_train_from_map(
    const spc_map_entry_t *entries, size_t entry_count,
    spc_bytes_t *out_model);

SPC_API spc_status_t spc_sentencepiece_trainer_train_from_map_with_sentences(
    const spc_map_entry_t *entries, size_t entry_count,
    const spc_string_view_t *sentences, size_t sentence_count,
    spc_bytes_t *out_model);

// -----------------------------------------------------------------------------
// SentencePieceNormalizer façade
// -----------------------------------------------------------------------------

typedef struct spc_sentencepiece_normalizer spc_sentencepiece_normalizer_t;

SPC_API spc_sentencepiece_normalizer_t *
spc_sentencepiece_normalizer_create(void);

SPC_API void spc_sentencepiece_normalizer_destroy(
    spc_sentencepiece_normalizer_t *normalizer);

SPC_API spc_status_t spc_sentencepiece_normalizer_load_from_file(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t filename);

SPC_API spc_status_t spc_sentencepiece_normalizer_load_from_serialized_proto(
    spc_sentencepiece_normalizer_t *normalizer, const void *data,
    size_t length);

SPC_API spc_status_t spc_sentencepiece_normalizer_load_from_rule_tsv(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t filename);

SPC_API spc_status_t spc_sentencepiece_normalizer_load_from_rule_name(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t name);

SPC_API spc_status_t spc_sentencepiece_normalizer_normalize(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t input,
    spc_bytes_t *out_text);

SPC_API spc_status_t spc_sentencepiece_normalizer_normalize_with_offsets(
    spc_sentencepiece_normalizer_t *normalizer, spc_string_view_t input,
    spc_normalized_result_t *out_result);

SPC_API spc_status_t spc_sentencepiece_normalizer_serialized_model_proto(
    spc_sentencepiece_normalizer_t *normalizer, spc_bytes_t *out_model);

// -----------------------------------------------------------------------------
// Global utilities
// -----------------------------------------------------------------------------

SPC_API void spc_set_random_generator_seed(uint32_t seed);
SPC_API void spc_set_min_log_level(int level);
SPC_API void spc_set_data_dir(spc_string_view_t data_dir);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SENTENCEPIECE_BINDINGS_C_SENTENCEPIECE_C_H_
