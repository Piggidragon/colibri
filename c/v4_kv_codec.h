#ifndef COLIBRI_V4_KV_CODEC_H
#define COLIBRI_V4_KV_CODEC_H

/* Encoded row storage for DeepSeek V4 attention.
 *
 * Main attention rows and Lightning Indexer rows use different native
 * checkpoint formats.  Keep the stream explicit: the tiny fixture uses the
 * same dimension for both, so inferring the layout from head_dim is unsafe.
 * Lossy index codecs remain independently selected through V4_KV_INDEX because
 * they can change the top-k token set, which is model semantics.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    COLI_V4_KV_F32 = 0,
    COLI_V4_KV_NATIVE,
    COLI_V4_KV_TURBO4,
    COLI_V4_KV_TURBO3,
    COLI_V4_KV_TURBO2,
} ColiV4KVCodec;

typedef enum {
    COLI_V4_KV_MAIN = 0,
    COLI_V4_KV_INDEX,
} ColiV4KVStream;

/* rope_dim is used only by main rows; index rows require zero.  Zero means the
 * requested layout cannot represent this geometry. */
size_t coli_v4_kv_row_bytes(ColiV4KVCodec codec, ColiV4KVStream stream,
                            int head_dim, int rope_dim);
int coli_v4_kv_encode_row(ColiV4KVCodec codec, ColiV4KVStream stream,
                          void *dst, const float *src,
                          int head_dim, int rope_dim);
int coli_v4_kv_decode_row(ColiV4KVCodec codec, ColiV4KVStream stream,
                          float *dst, const void *src,
                          int head_dim, int rope_dim);
float coli_v4_kv_dot(ColiV4KVCodec codec, ColiV4KVStream stream,
                     const float *query, const void *row,
                     int head_dim, int rope_dim);
void coli_v4_kv_accumulate(ColiV4KVCodec codec, ColiV4KVStream stream,
                           float *acc, float probability, const void *row,
                           int head_dim, int rope_dim);

const char *coli_v4_kv_codec_name(ColiV4KVCodec codec);
ColiV4KVCodec coli_v4_kv_codec_from_env(const char *variable,
                                        ColiV4KVStream stream,
                                        int head_dim, int rope_dim,
                                        ColiV4KVCodec fallback);

uint64_t coli_v4_kv_context_bytes(
    int layers, int sliding_window, int head_dim, int rope_dim,
    int index_head_dim, const int *compress_ratios, int context,
    ColiV4KVCodec codec, ColiV4KVCodec index_codec);

#ifdef __cplusplus
}
#endif

#endif
