#ifndef COLIBRI_DEEPSEEK_V4_INTERNAL_H
#define COLIBRI_DEEPSEEK_V4_INTERNAL_H

/*
 * Internal DeepSeek-V4 API. Not a stability commitment.
 * External callers should use deepseek_v4.h (engine / config / prompt).
 */
#include "deepseek_v4.h"

#include "tensor.h"
#include "expert_store.h"
#include "native_quant.h"
#include "native_quant_batch.h"
#include "native_quant_dual.h"
#include "native_quant_fp4_rows16.h"
#include "st.h"

#define COLI_ST_MAX_RANK ST_MAX_RANK
#define COLI_ST_BF16 0
#define COLI_ST_F16 1
#define COLI_ST_F32 2
#define COLI_ST_U8 3
#define COLI_ST_I8 3
#define COLI_ST_F8_E4M3 4
#define COLI_ST_F8_E8M0 5
#define COLI_ST_I64 6

typedef int ColiSafetensorsDType;
typedef st_tensor ColiSafetensorsTensor;
typedef shards ColiSafetensorsIndex;

typedef struct {
    ColiTensorView view;
    void *data_allocation;
    void *scale_allocation;
} ColiOwnedTensor;

typedef struct {
    float *data;
    uint64_t count;
    int rank;
    int64_t shape[COLI_ST_MAX_RANK];
} ColiFloatTensor;

int coli_st_index_open(ColiSafetensorsIndex **out, const char *directory,
                       char *error, size_t error_size);
void coli_st_index_close(ColiSafetensorsIndex *index);
size_t coli_st_tensor_count(const ColiSafetensorsIndex *index);
size_t coli_st_shard_count(const ColiSafetensorsIndex *index);
const char *coli_st_shard_path(const ColiSafetensorsIndex *index, int shard);
const ColiSafetensorsTensor *coli_st_find(const ColiSafetensorsIndex *index,
                                         const char *name);
int coli_st_tensor_shard(const ColiSafetensorsIndex *index,
                         const ColiSafetensorsTensor *tensor);
int coli_st_read_tensor(const ColiSafetensorsIndex *index,
                        const ColiSafetensorsTensor *tensor, void *destination);
int coli_st_read_at(const ColiSafetensorsIndex *index, int shard,
                    uint64_t offset, size_t length, void *destination);
/* Large, transient SSD read: prefer the index's O_DIRECT twin and use an
 * aligned bounce buffer, falling back to the ordinary buffered path. */
int coli_st_read_at_streaming(const ColiSafetensorsIndex *index, int shard,
                              uint64_t offset, size_t length,
                              void *destination);
int coli_st_streaming_direct_available(const ColiSafetensorsIndex *index,
                                       int shard);
int coli_st_prefetch_at(const ColiSafetensorsIndex *index, int shard,
                        uint64_t offset, size_t length);
const char *coli_st_dtype_name(ColiSafetensorsDType dtype);

int coli_tensor_load_fp8(ColiOwnedTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *prefix, char *error, size_t error_size);
void coli_owned_tensor_free(ColiOwnedTensor *tensor);
int coli_tensor_load_f32(ColiFloatTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *name, char *error, size_t error_size);
void coli_float_tensor_free(ColiFloatTensor *tensor);

typedef struct ColiV4Engine ColiV4Engine;

/* Runtime-selected full DSpark profile, shared with the separately compiled
 * generation unit. */
extern int coli_v4_full_dspark_wanted;
double coli_v4_dspark_cache_gb(void);

/* ==== begin deepseek_v4_math.h ==== */

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_hc_split_sinkhorn(float *pre, float *post, float *comb,
                              const float *mixes, const float scale[3],
                              const float *base, int hc, int iterations,
                              float eps);

int coli_v4_hc_pre(float *output, float *post, float *comb,
                   const float *input, const float *hc_fn,
                   const float scale[3], const float *base,
                   int hc, int dimension, int iterations,
                   float norm_eps, float hc_eps);

int coli_v4_hc_post(float *output, const float *branch,
                    const float *residual, const float *post,
                    const float *comb, int hc, int dimension);

int coli_v4_rmsnorm(float *output, const float *input, const float *weight,
                    int dimension, float eps);

int coli_v4_rope_precompute(float *cosines, float *sines,
                            int dimension, int sequence_length,
                            int original_sequence_length, float base,
                            float factor, int beta_fast, int beta_slow);

int coli_v4_rope_apply(float *vectors, int vector_count, int dimension,
                       const float *cosines, const float *sines, int inverse);

int coli_v4_route(float *weights, int *indices, const float *hidden,
                  const float *gate, const float *bias,
                  const int *forced_indices, int experts, int dimension,
                  int topk, float route_scale);

int coli_v4_swiglu(float *output, const float *gate, const float *up,
                   int dimension, float limit);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_math.h ==== */

/* Flash attention opt-out (V4_FLASH=0). Shared by every unit that carries a
 * copy of the attention source, so it lives here rather than in one unit. */
#include <stdlib.h>
#include <string.h>

static inline int v4_flash_enabled(void) {
    const char *setting = getenv("V4_FLASH");
    if (!setting || strcmp(setting, "1") == 0) return 1;
    if (strcmp(setting, "0") == 0) return 0;
    return 1;
}

#ifdef COLI_V4_CUDA
/* Defined once in COLI_V4_UNIT_MATH; latches the CPU-fallback warning so it is
 * printed once per process rather than once per attention unit. */
extern int coli_v4_attention_cuda_warned;
/* Also COLI_V4_UNIT_MATH; latches the resident-dense GPU path off after its
 * first failure, because the host FP8 buffers are gone by then. */
extern int coli_v4_dense_cuda_disabled;
#endif

/* ==== begin deepseek_v4_layer.h ==== */

#include <stddef.h>
#include <stdint.h>

#include "v4_kv_codec.h"
#ifdef COLI_V4_CUDA
#include "backend_cuda.h"
#endif

/* amalgamated: deepseek_v4_config.h */

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_V4_MAX_LAYER_TENSORS 48
#define COLI_V4_MAX_TENSOR_NAME 160

typedef struct {
    char name[COLI_V4_MAX_TENSOR_NAME];
    ColiSafetensorsDType dtype;
    int rank;
    int64_t shape[COLI_ST_MAX_RANK];
    /* Dense FP8 weights may be transposed inside each 8-row tile after load.
     * This is an in-memory execution layout only; checkpoint bytes and scales
     * remain unchanged. */
    int packed_rows8;
} ColiDeepSeekV4TensorSpec;

typedef struct {
    int layer;
    int compression_ratio;
    int uses_hash_router;
    int has_compressor;
    int has_indexer;
    size_t tensor_count;
    ColiDeepSeekV4TensorSpec tensors[COLI_V4_MAX_LAYER_TENSORS];
} ColiDeepSeekV4LayerPlan;

typedef struct {
    size_t tensor_count;
    uint64_t total_bytes;
    uint64_t bf16_bytes;
    uint64_t f32_bytes;
    uint64_t fp8_weight_bytes;
    uint64_t fp8_scale_bytes;
    uint64_t i64_bytes;
} ColiDeepSeekV4LayerStats;

typedef struct {
    ColiDeepSeekV4LayerPlan plan;
    ColiDeepSeekV4LayerStats stats;
    /* Written by the loader from the engine's dense location: it decides
     * whether the FP8 stays in checkpoint row-major order for the GPU.  Read it
     * afterwards to see whether the layer really landed in VRAM -- an upload
     * that fell back to the CPU clears it again. */
    int gpu_resident;
    void *data[COLI_V4_MAX_LAYER_TENSORS];
#ifdef COLI_V4_CUDA
    ColiCudaTensor *device[COLI_V4_MAX_LAYER_TENSORS];
    const ColiSafetensorsIndex *source_index;
    uint64_t host_bytes;
    uint64_t device_bytes;
#endif
} ColiDeepSeekV4LayerWeights;

int coli_v4_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                       const ColiDeepSeekV4Config *config, int layer,
                       char *error, size_t error_size);
int coli_v4_layer_validate(const ColiDeepSeekV4LayerPlan *plan,
                           const ColiSafetensorsIndex *index,
                           ColiDeepSeekV4LayerStats *stats,
                           char *error, size_t error_size);
int coli_v4_layer_load(ColiV4Engine *engine,
                       ColiDeepSeekV4LayerWeights *weights,
                       const ColiDeepSeekV4Config *config,
                       const ColiSafetensorsIndex *index, int layer,
                       char *error, size_t error_size);
void coli_v4_layer_free(ColiV4Engine *engine,
                        ColiDeepSeekV4LayerWeights *weights);
const void *coli_v4_layer_data(const ColiDeepSeekV4LayerWeights *weights,
                               const char *name,
                               const ColiDeepSeekV4TensorSpec **spec);
int coli_v4_dense_matmul(float *output,
                         const ColiDeepSeekV4LayerWeights *weights,
                         const char *prefix, const float *input, int batch,
                         int row_start, int rows);
int coli_v4_dense_shared_expert(
    float *output, const ColiDeepSeekV4LayerWeights *weights,
    const float *input, float swiglu_limit);
#ifdef COLI_V4_TEST_HOOKS
int coli_v4_test_fp8_maybe_pack_rows8(unsigned char *data,
                                      int64_t rows, int64_t columns,
                                      int gpu_resident);
#endif

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_layer.h ==== */

/* ==== begin deepseek_v4_sparse_attention.h ==== */

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_sparse_attention_ref(float *output, const float *queries,
                                 const float *kv, const float *sinks,
                                 const int *indices, int heads,
                                 int head_dimension, int kv_count, int topk,
                                 float softmax_scale);
int coli_v4_flash_attention_ref(
    float *output, const float *queries,
    const float *window_kv, int window_size,
    const float *compressed_kv, int compressed_count,
    const int *window_indices,
    const int *compressed_indices, int compressed_selected,
    const float *sinks, int heads, int head_dimension, float softmax_scale);
int coli_v4_flash_attention_codec_ref(
    float *output, const float *queries,
    const void *window_kv, int window_size,
    const void *compressed_kv, int compressed_count,
    const int *window_indices,
    const int *compressed_indices, int compressed_selected,
    ColiV4KVCodec codec, int rope_dimension,
    const float *sinks, int heads, int head_dimension, float softmax_scale);
int coli_v4_attention_two_source_ref(
    float *output, const float *queries,
    const float *window_kv, int window_size,
    const float *compressed_kv, int compressed_count,
    const int *window_indices,
    const int *compressed_indices, int compressed_selected,
    const float *sinks, int heads, int head_dimension, float softmax_scale);
int coli_v4_attention_two_source_codec_ref(
    float *output, const float *queries,
    const void *window_kv, int window_size,
    const void *compressed_kv, int compressed_count,
    const int *window_indices,
    const int *compressed_indices, int compressed_selected,
    ColiV4KVCodec codec, int rope_dimension,
    const float *sinks, int heads, int head_dimension, float softmax_scale);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_sparse_attention.h ==== */

/* ==== begin deepseek_v4_kv_cache.h ==== */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4KVCache ColiDeepSeekV4KVCache;

int coli_v4_kv_cache_create(ColiDeepSeekV4KVCache **cache,
                            int window_size, int compression_ratio,
                            int head_dimension, int max_context);
void coli_v4_kv_cache_reset(ColiDeepSeekV4KVCache *cache);
void coli_v4_kv_cache_destroy(ColiDeepSeekV4KVCache *cache);
int coli_v4_kv_cache_put_window(ColiDeepSeekV4KVCache *cache,
                                int position, const float *kv);
int coli_v4_kv_cache_put_compressed(ColiDeepSeekV4KVCache *cache,
                                    int position, const float *kv);
int coli_v4_kv_cache_indices(const ColiDeepSeekV4KVCache *cache,
                             int position, int *indices, size_t capacity);
const float *coli_v4_kv_cache_values(const ColiDeepSeekV4KVCache *cache);
int coli_v4_kv_cache_value_count(const ColiDeepSeekV4KVCache *cache);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_kv_cache.h ==== */

/* ==== begin deepseek_v4_attention_cache.h ==== */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4AttentionCache ColiDeepSeekV4AttentionCache;

int coli_v4_attention_cache_create(ColiDeepSeekV4AttentionCache **cache,
                                   int window_size, int compression_ratio,
                                   int head_dimension, int max_context);
void coli_v4_attention_cache_reset(ColiDeepSeekV4AttentionCache *cache);
void coli_v4_attention_cache_destroy(ColiDeepSeekV4AttentionCache *cache);

/* query is [heads, head_dimension]. window_kv and compressed_kv have one
 * head_dimension vector each. compressed_kv is required at ratio boundaries. */
int coli_v4_attention_cache_step(ColiDeepSeekV4AttentionCache *cache,
                                 float *output, const float *query,
                                 const float *window_kv,
                                 const float *compressed_kv,
                                 const float *sinks, int heads,
                                 int position, float softmax_scale);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_attention_cache.h ==== */

/* ==== begin deepseek_v4_attention.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4WindowAttentionState
    ColiDeepSeekV4WindowAttentionState;

int coli_v4_window_attention_create(ColiDeepSeekV4WindowAttentionState **state,
                                    const ColiDeepSeekV4Config *config,
                                    ColiV4KVCodec codec,
                                    ColiV4KVCodec index_codec);
void coli_v4_window_attention_reset(ColiDeepSeekV4WindowAttentionState *state);
void coli_v4_window_attention_destroy(ColiDeepSeekV4WindowAttentionState *state);

/* Correctness-first single-KV attention. Compressed layers may use this at
 * position zero, before any compressed KV/indexer candidate exists. */
int coli_v4_attention_token_ref(float *output,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size);
int coli_v4_attention_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_attention.h ==== */

/* ==== begin deepseek_v4_attention_batch.h ==== */

/* amalgamated: deepseek_v4_attention.h */

int coli_v4_attention_window_batch_ref(
    float *outputs, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int start_position, int batch, char *error, size_t error_size);
/* ==== end deepseek_v4_attention_batch.h ==== */

/* ==== begin deepseek_v4_attention_transaction.h ==== */

/* amalgamated: deepseek_v4_attention.h */

typedef struct ColiV4AttentionSnapshot ColiV4AttentionSnapshot;

int coli_v4_attention_snapshot_create(
    const ColiDeepSeekV4WindowAttentionState *state,
    ColiV4AttentionSnapshot **output);
int coli_v4_attention_snapshot_restore(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiV4AttentionSnapshot *snapshot);
void coli_v4_attention_snapshot_destroy(ColiV4AttentionSnapshot *snapshot);
#ifdef COLI_V4_TEST_HOOKS
int coli_v4_test_attention_encode_rejection_error(void);
int coli_v4_test_attention_snapshot_roundtrip(ColiV4KVCodec codec);
int coli_v4_test_indexer_snapshot_rejections(void);
#endif
/* ==== end deepseek_v4_attention_transaction.h ==== */

/* ==== begin deepseek_v4_compressor.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4CompressorState ColiDeepSeekV4CompressorState;

typedef struct {
    const char *prefix;
    int head_dimension;
    int rotate_fp4;
} ColiDeepSeekV4CompressorOptions;

int coli_v4_compressor_create(ColiDeepSeekV4CompressorState **state,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              char *error, size_t error_size);
int coli_v4_compressor_create_with_options(
    ColiDeepSeekV4CompressorState **state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size);
void coli_v4_compressor_reset(ColiDeepSeekV4CompressorState *state);
int coli_v4_compressor_bind_weights(ColiDeepSeekV4CompressorState *state,
                                    const ColiDeepSeekV4LayerWeights *weights,
                                    char *error, size_t error_size);
void coli_v4_compressor_destroy(ColiDeepSeekV4CompressorState *state);

/* Processes one decode token. produced is set to one only when a complete
 * compression window emits a KV vector. output may be NULL on other steps. */
int coli_v4_compressor_step(ColiDeepSeekV4CompressorState *state,
                            float *output, int *produced,
                            const float *input, int position,
                            char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_compressor.h ==== */

/* ==== begin deepseek_v4_compressor_snapshot.h ==== */

/* amalgamated: deepseek_v4_compressor.h */

typedef struct ColiV4CompressorSnapshot ColiV4CompressorSnapshot;

int coli_v4_compressor_snapshot_create(
    const ColiDeepSeekV4CompressorState *state,
    ColiV4CompressorSnapshot **output);
int coli_v4_compressor_snapshot_restore(
    ColiDeepSeekV4CompressorState *state,
    const ColiV4CompressorSnapshot *snapshot);
void coli_v4_compressor_snapshot_destroy(ColiV4CompressorSnapshot *snapshot);
/* ==== end deepseek_v4_compressor_snapshot.h ==== */

/* ==== begin deepseek_v4_indexer.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */

typedef struct ColiDeepSeekV4Indexer ColiDeepSeekV4Indexer;

int coli_v4_indexer_create(ColiDeepSeekV4Indexer **state,
                           const ColiDeepSeekV4LayerWeights *weights,
                           const ColiDeepSeekV4Config *config,
                           int max_context, ColiV4KVCodec codec,
                           char *error, size_t error_size);
int coli_v4_indexer_bind_weights(ColiDeepSeekV4Indexer *state,
                                 const ColiDeepSeekV4LayerWeights *weights,
                                 char *error, size_t error_size);
void coli_v4_indexer_reset(ColiDeepSeekV4Indexer *state);
void coli_v4_indexer_destroy(ColiDeepSeekV4Indexer *state);

/* Updates the overlap compressor, then returns compressed-cache ordinals in
 * descending index score order. query_rank is the normalized q_lora vector. */
int coli_v4_indexer_step(ColiDeepSeekV4Indexer *state, int *indices,
                         int index_capacity, const float *query_rank,
                         const float *input, int position,
                         char *error, size_t error_size);
const void *coli_v4_indexer_compressed_values(
    const ColiDeepSeekV4Indexer *state);
int coli_v4_indexer_compressed_count(const ColiDeepSeekV4Indexer *state);
/* ==== end deepseek_v4_indexer.h ==== */

/* ==== begin deepseek_v4_indexer_snapshot.h ==== */

/* amalgamated: deepseek_v4_indexer.h */

typedef struct ColiV4IndexerSnapshot ColiV4IndexerSnapshot;

int coli_v4_indexer_snapshot_create(const ColiDeepSeekV4Indexer *state,
                                    ColiV4IndexerSnapshot **output);
int coli_v4_indexer_snapshot_restore(ColiDeepSeekV4Indexer *state,
                                     const ColiV4IndexerSnapshot *snapshot);
void coli_v4_indexer_snapshot_destroy(ColiV4IndexerSnapshot *snapshot);
/* ==== end deepseek_v4_indexer_snapshot.h ==== */

/* ==== begin deepseek_v4_expert.h ==== */

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_expert_forward_ref(float *output, const ColiExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit);

int coli_v4_shared_expert_forward_ref(float *output,
                                      const ColiTensorView *gate,
                                      const ColiTensorView *down,
                                      const ColiTensorView *up,
                                      const float *input,
                                      float swiglu_limit);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_expert.h ==== */

/* ==== begin deepseek_v4_expert_store.h ==== */

#include <stddef.h>
#include <stdint.h>

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *model_dir;
    int layers;
    int experts_per_layer;
    uint64_t cache_bytes;
    /* Optional hot-pin policy (-1 / 0 => implementation default). */
    int pin_slots_per_layer;
    uint64_t repin_interval;
} ColiDeepSeekV4ExpertStoreOptions;

int coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_expert_store.h ==== */

/* ==== begin deepseek_v4_block.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_attention.h */
/* amalgamated: deepseek_v4_layer.h */
#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_block_token_ref(float *output_hc,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size);
int coli_v4_block_window_token_ref(
    float *output_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_block.h ==== */

/* ==== begin deepseek_v4_block_batch.h ==== */

/* amalgamated: deepseek_v4_attention.h */
/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */
#include "expert_store.h"

int coli_v4_block_window_batch_ref(
    float *outputs_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens, int start_position, int batch,
    char *error, size_t error_size);
/* ==== end deepseek_v4_block_batch.h ==== */

/* ==== begin deepseek_v4_resource_plan.h ==== */

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t available_bytes;
    uint64_t user_limit_bytes;
    uint64_t maximum_layer_bytes;
    uint64_t runtime_other_bytes;
    uint64_t expert_record_bytes;
    int sparse_layers;
    int routed_topk;
    int experts_per_layer;
} ColiDeepSeekV4ResourceInputs;

typedef struct {
    uint64_t os_available_bytes;
    uint64_t planner_available_bytes;
    uint64_t system_reserve_bytes;
    uint64_t runtime_reserve_bytes;
    uint64_t minimum_expert_bytes;
    uint64_t expert_cache_bytes;
    uint64_t projected_bytes;
    int slots_per_layer;
} ColiDeepSeekV4ResourcePlan;

typedef struct {
    uint64_t available_bytes;
    uint64_t vram_available_bytes;
    /* Kept clear of the dense tier: the per-session KV mirror and the kernel
     * scratch are allocated later and have nowhere else to go. */
    uint64_t vram_reserve_bytes;
    uint64_t fixed_bytes;
    uint64_t dense_bytes;
    uint64_t dense_device_bytes;
    uint64_t minimum_expert_bytes;
    int vram_enabled;
} ColiDeepSeekV4ResidentTierInputs;

typedef enum {
    COLI_V4_DENSE_STREAMED = 0,
    COLI_V4_DENSE_RAM = 1,
    COLI_V4_DENSE_VRAM = 2,
} ColiDeepSeekV4DenseLocation;

typedef struct {
    ColiDeepSeekV4DenseLocation dense_location;
    uint64_t dense_bytes;
    uint64_t dense_device_bytes;
    int dense_resident;
} ColiDeepSeekV4ResidentTierPlan;

uint64_t coli_v4_os_available_memory(void);
uint64_t coli_v4_scratch_bytes(void);
int coli_v4_context_tokens(void);
/* V4_VRAM_LIMIT_MB: caps what the planner believes is free, without owning a
 * smaller card.  Clamps down only -- a value at or above `free_bytes`, or
 * malformed/unset input, leaves `free_bytes` unchanged.  `0` is a valid
 * clamp-down value (not malformed) and forces the result to zero. */
uint64_t coli_v4_vram_limit_bytes(uint64_t free_bytes);
/* V4_PIN_SLOTS (absolute) / V4_PIN_FRACTION (share of available_pins) turn
 * the former COLI_V4_MAX_PIN_SLOTS_PER_LAYER compile-time ceiling into a
 * runtime one.  V4_PIN_SLOTS wins if both are set.  Result is always
 * clamped into [0, available_pins]; garbage/out-of-range input on either
 * knob falls back to the compile-time default. */
int coli_v4_pin_slots_ceiling(int available_pins);
/* V4_PIN_RAMP_REQUESTS overrides the compile-time COLI_V4_PIN_RAMP_REQUESTS
 * default (0 = ramp disabled).  Garbage/negative input falls back to the
 * compile-time default. */
int coli_v4_pin_ramp_requests(void);

/* Shared by both indexer copies (INDEXER, INDEXER_SNAPSHOT -- see
 * plans/12-expert-cache-policy.md Commit 2) and their test. Defined here as
 * static inline, like coli_v4_vram_fail_at() above, rather than living in
 * either unit: the function is pure array logic with no v4 state, and a
 * unit-local definition would need the rename-macro trick every other
 * INDEXER_SNAPSHOT symbol uses to avoid a duplicate-symbol link error, for
 * no benefit -- this way each translation unit that includes this header
 * (including the test, with no deepseek_v4.c dependency at all) gets its
 * own private copy. */
typedef struct { float score; int index; } ColiV4IndexScore;

/* True if `a` sorts strictly after `b`: lower score first, ties broken by
 * ascending index. Mirrors deepseek_v4.c's descending_score() qsort
 * comparator exactly (same total order), so coli_v4_indexer_select()'s
 * output for the retained set is byte-identical to what
 * qsort(scores, count, sizeof(*scores), descending_score) would have put in
 * scores[0..topk). */
static inline int coli_v4_index_score_worse(const ColiV4IndexScore *a,
                                            const ColiV4IndexScore *b) {
    /* Mirrors the removed descending_score()'s branch structure, not just
     * its result on finite scores: `<` and `>` are both false whenever
     * either operand is NaN, so NaN falls into the same index tiebreak that
     * real ties use, exactly like the old qsort comparator did. A `!=`-based
     * test would instead treat any NaN pair as "not equal" and skip the
     * tiebreak, making two distinct NaN entries compare as equal ranks. */
    if (a->score < b->score) return 1;
    if (a->score > b->score) return 0;
    return a->index > b->index;
}

static inline int coli_v4_index_score_descending(const void *left,
                                                  const void *right) {
    const ColiV4IndexScore *a = left, *b = right;
    if (coli_v4_index_score_worse(a, b)) return 1;
    if (coli_v4_index_score_worse(b, a)) return -1;
    return 0;
}

/* Min-heap sift-down: heap[0..size) is a min-heap by "worse" (root is the
 * single worst-ranked element of the retained set, so a better candidate
 * can evict it in O(log size) instead of re-sorting). */
static inline void coli_v4_index_score_sift_down(
    ColiV4IndexScore *heap, int size, int root) {
    for (;;) {
        int left = 2 * root + 1, right = left + 1, smallest = root;
        if (left < size &&
            coli_v4_index_score_worse(&heap[left], &heap[smallest]))
            smallest = left;
        if (right < size &&
            coli_v4_index_score_worse(&heap[right], &heap[smallest]))
            smallest = right;
        if (smallest == root) return;
        ColiV4IndexScore swap = heap[root];
        heap[root] = heap[smallest]; heap[smallest] = swap;
        root = smallest;
    }
}

/* Selects the best `topk` of scores[0..count), by descending score with
 * ties broken by ascending index, into scores[0..min(topk,count)) in that
 * same sorted order -- the identical set and order a full
 * qsort(..., descending_score) would produce, without sorting the discarded
 * remainder (O(count log topk) instead of O(count log count), see
 * plans/12-expert-cache-policy.md Commit 2). count < 0 is treated as 0;
 * topk <= 0 is a no-op. Returns min(topk, count). Router semantics: the
 * selected set and order must never differ from a full descending qsort --
 * see test_v4_indexer_select.c. */
static inline int coli_v4_indexer_select(
    ColiV4IndexScore *scores, int count, int topk) {
    if (count < 0) count = 0;
    if (topk > count) topk = count;
    if (topk <= 0) return 0;
    if (topk < count) {
        for (int i = topk / 2 - 1; i >= 0; i--)
            coli_v4_index_score_sift_down(scores, topk, i);
        for (int i = topk; i < count; i++) {
            if (coli_v4_index_score_worse(&scores[0], &scores[i])) {
                scores[0] = scores[i];
                coli_v4_index_score_sift_down(scores, topk, 0);
            }
        }
    }
    qsort(scores, (size_t)topk, sizeof(*scores),
          coli_v4_index_score_descending);
    return topk;
}

int coli_v4_session_state_bytes(int context_tokens, int hc_mult,
                                int hidden_size, uint64_t *bytes);
/* V4_PREFILL_CHUNK=0 preserves the full-prompt allocation.  A valid value is
 * clamped to the V4 chunk contract and then capped by this session's prompt
 * capacity, so callers can use the answer directly as a buffer length. */
int coli_v4_prefill_chunk_tokens(int max_prompt_tokens);
int coli_v4_resource_plan_compute(
    ColiDeepSeekV4ResourcePlan *plan,
    const ColiDeepSeekV4ResourceInputs *inputs,
    char *error, size_t error_size);
int coli_v4_resident_tier_plan(
    ColiDeepSeekV4ResidentTierPlan *plan,
    const ColiDeepSeekV4ResidentTierInputs *inputs,
    char *error, size_t error_size);

/* plans/08-vram-planner.md: which of the four VRAM-eligible tiers actually
 * fit the card, in fixed priority order.  This is the VRAM-side counterpart
 * to coli_v4_resident_tier_plan above -- it never decides RAM-vs-streamed,
 * only VRAM-vs-not-VRAM, because the RAM fallback for each tier is already
 * decided elsewhere (coli_v4_resident_tier_plan for dense, the 256 MiB head
 * margin check for head, the RAM-lazy path for DSpark). RAM is always a
 * legal fallback for kv/dense/head; dspark's "not VRAM" location means the
 * existing RAM-lazy path, not an error. */
typedef enum {
    COLI_V4_TIER_VRAM = 0,
    COLI_V4_TIER_RAM = 1,
} ColiV4TierLocation;

typedef struct {
    uint64_t vram_available_bytes;   /* raw coli_cuda_mem_info free bytes */
    uint64_t vram_reserve_bytes;     /* driver/allocator headroom, off the top */
    uint64_t kv_bytes;               /* attention main+compressed rows only */
    uint64_t dense_bytes;            /* device-eligible FP8 weights+scales */
    uint64_t head_bytes;
    uint64_t dspark_bytes;
    int dspark_wanted;               /* 0: dspark is off, never claims VRAM */
} ColiV4VramTierInputs;

typedef struct {
    ColiV4TierLocation kv, dense, head, dspark;
    uint64_t vram_used_bytes;        /* sum of what actually landed on the card */
} ColiV4VramTierPlan;

int coli_v4_vram_tier_plan(
    ColiV4VramTierPlan *plan, const ColiV4VramTierInputs *inputs,
    char *error, size_t error_size);

/* V4_VRAM_FAIL_AT=kv|dense|head|dspark: force that stage's device upload to
 * fail, so the plan's runtime-degrade path (planning said VRAM, the upload
 * fails anyway) can be exercised without owning a card small enough to fail
 * for real.  Not compiled into production objects: outside COLI_V4_TEST_HOOKS
 * this is a constant fold to "never", so it costs nothing and cannot be
 * reached via the environment in a release build. */
typedef enum {
    COLI_V4_VRAM_FAIL_NONE = 0,
    COLI_V4_VRAM_FAIL_KV,
    COLI_V4_VRAM_FAIL_DENSE,
    COLI_V4_VRAM_FAIL_HEAD,
    COLI_V4_VRAM_FAIL_DSPARK,
} ColiV4VramFailStage;

#ifdef COLI_V4_TEST_HOOKS
ColiV4VramFailStage coli_v4_vram_fail_at(void);
#else
static inline ColiV4VramFailStage coli_v4_vram_fail_at(void) {
    return COLI_V4_VRAM_FAIL_NONE;
}
#endif
/* ==== end deepseek_v4_resource_plan.h ==== */

/* ==== begin deepseek_v4_head_cache.h ==== */

#include <stddef.h>
#include <stdint.h>

int coli_v4_head_cache_probe(const char *model_dir, uint64_t *bytes,
                             char *error, size_t error_size);
/* device=1 keeps the BF16 payload exclusively on CUDA.  A device allocation
 * failure returns 1 so the RAM tier planner can retry with a host head. */
int coli_v4_head_cache_load(ColiV4Engine *engine, const char *model_dir,
                            int device, char *error, size_t error_size);
uint64_t coli_v4_head_cache_bytes(const ColiV4Engine *engine);
const void *coli_v4_head_cache_data(const ColiV4Engine *engine,
                                    int shard, uint64_t offset, size_t length);
const void *coli_v4_head_cache_device(const ColiV4Engine *engine,
                                      int shard, uint64_t offset, size_t length);
void coli_v4_dspark_gpu_release(void);
/* ==== end deepseek_v4_head_cache.h ==== */


/* Runtime options live on ColiV4Engine. */
typedef struct {
    const char *target_model_dir;
    uint64_t memory_limit_bytes;
    int context_tokens;
    int dense_resident;
    ColiDeepSeekV4DenseLocation dense_location;
    uint64_t target_expert_cache_bytes;
    int pin_slots_per_layer;
    uint64_t repin_interval;
    uint64_t dspark_reserve_bytes;
    ColiV4KVCodec kv_codec;
    ColiV4KVCodec index_codec;
    int vram_enabled;
    /* plans/08-vram-planner.md: whether the KV tier plan (computed once, at
     * expert-store-open time) admitted the attention main+compressed rows to
     * the device.  window_attention_create gates its device mirror on this
     * instead of the bare v4_cuda_ready() check phase 05 shipped with, so
     * dense/head/dspark's earlier device claims cannot silently starve every
     * session's KV mirror of the budget the planner meant to give it. */
    int kv_vram_enabled;
    /* plans/08-vram-planner.md: whether the VRAM tier plan actually admitted
     * the DSpark backbone to the device (vram_plan.dspark == VRAM).  The
     * lazy loader (v4_ds_load_all) gates its GPU upload on this instead of
     * the bare vram_enabled flag, so a plan that had to fall the drafter
     * back to RAM/host -- because KV, dense, or head already claimed the
     * budget it needed -- does not still attempt (and OOM on) the upload. */
    int dspark_vram_enabled;
} ColiDeepSeekV4RuntimeOptions;

enum { COLI_V4_RESIDENT_MAX_LAYERS = 128 };

struct ColiV4Engine {
    ColiDeepSeekV4Config config;
    ColiDeepSeekV4RuntimeOptions runtime;
    ColiSafetensorsIndex *target_index;
    ColiExpertStore *experts;
    ColiV4EngineMemorySummary summary;
    struct {
        unsigned char *data;
        void *device;
        uint64_t bytes;
        uint64_t offset;
        int shard;
    } head_cache;
    struct {
        ColiDeepSeekV4LayerWeights layers[COLI_V4_RESIDENT_MAX_LAYERS];
        unsigned char ready[COLI_V4_RESIDENT_MAX_LAYERS];
        const ColiSafetensorsIndex *index;
        uint64_t host_bytes;
        uint64_t device_bytes;
    } dense_resident;
    struct {
        uint16_t *markov_w1;
        uint16_t *markov_w2;
        uint64_t bytes;
        int rank;
        int block_size;
        int stage;
        int enabled;
    } dspark;
    char *owned_target_model_dir;
    int owns_experts;
    int owns_index;
    int active_sessions; /* sessions created against this engine */
};

/* Session ownership helpers shared by production session code and tests. */
void coli_v4_engine_attach_session(ColiV4Engine *engine);
void coli_v4_engine_detach_session(ColiV4Engine *engine);

#include "tok.h"
#include "kv_prefix.h"

struct ColiV4Session {
    ColiV4Engine *engine;
    ColiDeepSeekV4Config config;
    ColiDeepSeekV4WindowAttentionState **attention;
    float *state;
    float *next;
    float *hidden;
    int *prompt_ids;
    int *generated;
    int max_prompt_tokens;
    int prefill_chunk_tokens; /* 0 => full prompt, otherwise buffer capacity */
    int max_new_tokens_cap;
    int prompt_count;
    int generated_count;
    Tok tokenizer;
    int tokenizer_ready;
    char *text;
    int text_length;
    /* Token ids this session's attention state already holds, prompt and
     * generated alike, in the shared format colibri.c/inkling.c/kimi_k3.c use.
     * A follow-up request whose prompt starts with exactly these ids continues
     * from that position instead of re-prefilling it. */
    kv_prefix fed;
    int prefix_reused;   /* reuse length of the request in flight, for stats */
    uint64_t spec_attempts;
    uint64_t spec_drafted;
    uint64_t spec_accepted;
    int spec_disabled;
};

/* RAM-tiered expert open used by coli_v4_engine_open (replaces ld --wrap). */
int coli_v4_expert_store_open_planned(
    ColiV4Engine *engine,
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size);

/* Internal accessors — not part of the experimental public API. */
ColiSafetensorsIndex *coli_v4_engine_target_index(ColiV4Engine *engine);
ColiExpertStore *coli_v4_engine_expert_store(ColiV4Engine *engine);

/* Head-cache aware safetensors read (engine NULL => plain coli_st_read_at). */
int coli_st_read_at_engine(ColiV4Engine *engine,
                           const ColiSafetensorsIndex *index, int shard,
                           uint64_t offset, size_t length, void *destination);

#ifdef COLI_V4_TEST_HOOKS
/*
 * Fault-injection / counters for ownership tests only.
 * Compile ownership objects with -DCOLI_V4_TEST_HOOKS; production objects omit this.
 */
extern int coli_v4_test_fail_expert_store_open;
extern int coli_v4_test_skip_expert_store_open;
extern int coli_v4_test_closed_owned_index;

ColiV4Session *coli_v4_test_session_bare_create(ColiV4Engine *engine);
void coli_v4_test_session_bare_destroy(ColiV4Session *session);
#endif /* COLI_V4_TEST_HOOKS */

#endif /* COLIBRI_DEEPSEEK_V4_INTERNAL_H */
