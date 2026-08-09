#include "../backend_cuda_v4.h"
#include "../native_quant.h"
#include "../v4_kv_codec.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" int coli_v4_flash_attention_codec_ref(
    float *output, const float *queries,
    const void *window_kv, int window_size,
    const void *compressed_kv, int compressed_count,
    const int *window_indices,
    const int *compressed_indices, int compressed_selected,
    ColiV4KVCodec codec, int rope_dimension,
    const float *sinks, int heads, int head_dimension, float softmax_scale);

static uint32_t rng_state = UINT32_C(0x31415926);

static float random_signed(void) {
    rng_state = rng_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return ((float)((rng_state >> 8) & 0xffffu) / 32768.0f - 1.0f) * 0.75f;
}

static int make_model_row(float *row, int row_id) {
    float qdq[448];
    uint8_t scales[7];
    for (int i = 0; i < 512; i++)
        row[i] = random_signed() + 0.01f * sinf((float)(row_id + i));
    if (coli_fp8_activation_qdq_ref(qdq, scales, row, 448, 64)) return -1;
    memcpy(row, qdq, sizeof(qdq));
    coli_bf16_round_array(row, 448);
    coli_bf16_round_array(row + 448, 64);
    return 0;
}

static int close_enough(const float *reference, const float *actual,
                        size_t count, float *cosine) {
    double dot = 0.0, ref2 = 0.0, got2 = 0.0;
    for (size_t i = 0; i < count; i++) {
        float limit = 1e-3f + 8e-3f * fmaxf(fabsf(reference[i]),
                                             fabsf(actual[i]));
        if (fabsf(reference[i] - actual[i]) > limit) {
            fprintf(stderr,
                    "component %zu differs: cpu=%g cuda=%g limit=%g\n",
                    i, reference[i], actual[i], limit);
            return -1;
        }
        dot += (double)reference[i] * actual[i];
        ref2 += (double)reference[i] * reference[i];
        got2 += (double)actual[i] * actual[i];
    }
    *cosine = ref2 > 0.0 && got2 > 0.0
        ? (float)(dot / sqrt(ref2 * got2)) : 1.0f;
    return *cosine > 0.9999f ? 0 : -1;
}

static int run_case(ColiV4KVCodec codec, int heads,
                    int compressed_selected) {
    enum { HEAD_DIM = 512, ROPE_DIM = 64, WINDOW = 8 };
    int compressed_count = compressed_selected > 0 ? compressed_selected : 0;
    size_t row_bytes = coli_v4_kv_row_bytes(
        codec, COLI_V4_KV_MAIN, HEAD_DIM, ROPE_DIM);
    size_t host_rows = (size_t)WINDOW + compressed_count;
    float *rows = (float *)malloc(host_rows * HEAD_DIM * sizeof(float));
    unsigned char *window = (unsigned char *)malloc((size_t)WINDOW * row_bytes);
    unsigned char *compressed = compressed_count
        ? (unsigned char *)malloc((size_t)compressed_count * row_bytes) : NULL;
    float *queries = (float *)malloc((size_t)heads * HEAD_DIM * sizeof(float));
    float *cpu = (float *)malloc((size_t)heads * HEAD_DIM * sizeof(float));
    float *gpu = (float *)malloc((size_t)heads * HEAD_DIM * sizeof(float));
    float *sinks = (float *)malloc((size_t)heads * sizeof(float));
    int window_indices[WINDOW];
    int *compressed_indices = compressed_count
        ? (int *)malloc((size_t)compressed_count * sizeof(int)) : NULL;
    if (!row_bytes || !rows || !window || (compressed_count && !compressed) ||
        !queries || !cpu || !gpu || !sinks ||
        (compressed_count && !compressed_indices))
        return -1;

    for (size_t row = 0; row < host_rows; row++) {
        float *source = rows + row * HEAD_DIM;
        if (make_model_row(source, (int)row) || coli_v4_kv_encode_row(
                codec, COLI_V4_KV_MAIN,
                row < WINDOW ? window + row * row_bytes
                             : compressed + (row - WINDOW) * row_bytes,
                source, HEAD_DIM, ROPE_DIM))
            return -1;
    }
    for (int i = 0; i < WINDOW; i++)
        window_indices[i] = i == 3 ? -1 : i;
    for (int i = 0; i < compressed_count; i++)
        compressed_indices[i] = i && i % 257 == 0 ? -1 : i;
    for (int head = 0; head < heads; head++) {
        sinks[head] = heads == 1 && compressed_selected == 7
            ? 80.0f : (float)(head % 7 - 3) * 0.125f;
        for (int column = 0; column < HEAD_DIM; column++)
            queries[(size_t)head * HEAD_DIM + column] = random_signed();
    }

    void *device_window = v4_cuda_kv_alloc((size_t)WINDOW * row_bytes);
    void *device_compressed = compressed_count
        ? v4_cuda_kv_alloc((size_t)compressed_count * row_bytes) : NULL;
    if (!device_window || (compressed_count && !device_compressed)) return -1;
    for (int row = 0; row < WINDOW; row++)
        if (v4_cuda_kv_write_row(
                device_window, row, window + (size_t)row * row_bytes,
                row_bytes)) return -1;
    for (int row = 0; row < compressed_count; row++)
        if (v4_cuda_kv_write_row(
                device_compressed, row,
                compressed + (size_t)row * row_bytes, row_bytes)) return -1;

    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    const int *selected_indices = heads == 8 && compressed_selected == 64
        ? NULL : compressed_indices;
    int result = coli_v4_flash_attention_codec_ref(
        cpu, queries, window, WINDOW, compressed, compressed_count,
        window_indices, selected_indices, compressed_selected,
        codec, ROPE_DIM, sinks, heads, HEAD_DIM, scale);
    if (!result) result = v4_cuda_flash_attention(
        gpu, queries, device_window, WINDOW, window_indices,
        device_compressed, compressed_count, selected_indices,
        compressed_selected, sinks, codec, heads, HEAD_DIM, row_bytes, scale);
    float cosine = 0.0f;
    if (!result) result = close_enough(
        cpu, gpu, (size_t)heads * HEAD_DIM, &cosine);
    if (result)
        fprintf(stderr, "failed codec=%s heads=%d selected=%d cosine=%g\n",
                coli_v4_kv_codec_name(codec), heads,
                compressed_selected, cosine);

    v4_cuda_kv_free(device_compressed);
    v4_cuda_kv_free(device_window);
    free(compressed_indices); free(sinks); free(gpu); free(cpu); free(queries);
    free(compressed); free(window); free(rows);
    return result;
}

static int test_rejections_before_tables(void) {
    float row[512] = {0}, query[512] = {0}, output[512], sink = 0.0f;
    int index = 0;
    void *device = v4_cuda_kv_alloc(sizeof(row));
    if (!device || v4_cuda_kv_write_row(device, 0, row, sizeof(row))) return -1;
    int result = v4_cuda_flash_attention(
        output, query, device, 1, &index, NULL, 0, NULL, 0,
        &sink, COLI_V4_KV_F32, 1, 512, sizeof(row),
        1.0f / sqrtf(512.0f));
    v4_cuda_kv_free(device);
    return result == -1 ? 0 : -1;
}

static int test_invalid_compressed_index(void) {
    float row[512] = {0}, query[512] = {0}, output[512], sink = 0.0f;
    int window_index = 0, compressed_index = 1;
    void *window = v4_cuda_kv_alloc(sizeof(row));
    void *compressed = v4_cuda_kv_alloc(sizeof(row));
    if (!window || !compressed ||
        v4_cuda_kv_write_row(window, 0, row, sizeof(row)) ||
        v4_cuda_kv_write_row(compressed, 0, row, sizeof(row))) return -1;
    int result = v4_cuda_flash_attention(
        output, query, window, 1, &window_index, compressed, 1,
        &compressed_index, 1, &sink, COLI_V4_KV_F32, 1, 512,
        sizeof(row), 1.0f / sqrtf(512.0f));
    v4_cuda_kv_free(compressed);
    v4_cuda_kv_free(window);
    return result == -1 ? 0 : -1;
}

int main(void) {
    if (v4_cuda_init(0)) {
        puts("test_v4_attention_cuda: skipped (no CUDA device)");
        return 77;
    }
    if (test_rejections_before_tables() || v4_cuda_publish_tables() ||
        test_invalid_compressed_index())
        return 1;
    if (v4_cuda_kv_alloc(SIZE_MAX / 2) != NULL) {
        fprintf(stderr, "absurd allocation unexpectedly succeeded\n");
        return 1;
    }

    const ColiV4KVCodec codecs[] = {
        COLI_V4_KV_F32, COLI_V4_KV_NATIVE, COLI_V4_KV_TURBO4,
        COLI_V4_KV_TURBO3, COLI_V4_KV_TURBO2,
    };
    const int head_counts[] = {1, 8, 64};
    const int selections[] = {1, 7, 64, 2048};
    for (size_t codec = 0; codec < sizeof(codecs) / sizeof(codecs[0]); codec++)
        for (size_t heads = 0;
             heads < sizeof(head_counts) / sizeof(head_counts[0]); heads++)
            for (size_t selected = 0;
                 selected < sizeof(selections) / sizeof(selections[0]); selected++)
                if (run_case(codecs[codec], head_counts[heads],
                             selections[selected]))
                    return 1;
    v4_cuda_shutdown();
    puts("test_v4_attention_cuda: ok");
    return 0;
}
