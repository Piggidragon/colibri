#include "../deepseek_v4_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t random_state = UINT32_C(0x91e10da5);

static float random_float(void) {
    random_state = random_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return ((float)(random_state >> 8) / 16777216.0f - 0.5f) * 0.5f;
}

static void set_flash(const char *value);

static int compare_outputs(const float *reference, const float *flash,
                           size_t count, const char *label) {
    double dot = 0.0, reference_norm = 0.0, flash_norm = 0.0;
    for (size_t i = 0; i < count; i++) {
        float scale = fmaxf(fabsf(reference[i]), fabsf(flash[i]));
        float difference = fabsf(reference[i] - flash[i]);
        float tolerance = 1e-3f + 1e-3f * scale;
        if (difference > tolerance) {
            fprintf(stderr,
                    "%s[%zu]: reference=%g flash=%g difference=%g "
                    "tolerance=%g\n",
                    label, i, reference[i], flash[i], difference, tolerance);
            return 1;
        }
        dot += (double)reference[i] * flash[i];
        reference_norm += (double)reference[i] * reference[i];
        flash_norm += (double)flash[i] * flash[i];
    }
    double cosine = dot / sqrt(reference_norm * flash_norm);
    if (!(cosine > 0.9999)) {
        fprintf(stderr, "%s: cosine=%g\n", label, cosine);
        return 1;
    }
    return 0;
}

static int compare_random_case(int topk, int padded, int flash_enabled) {
    enum { HEADS = 8, HEAD_DIM = 512, WINDOW_LIMIT = 128 };
    int window_size = topk < WINDOW_LIMIT ? topk : WINDOW_LIMIT;
    int compressed_selected = topk - window_size;
    size_t values = (size_t)topk * HEAD_DIM;
    float *kv = malloc(values * sizeof(*kv));
    float *queries = malloc((size_t)HEADS * HEAD_DIM * sizeof(*queries));
    float *reference = malloc((size_t)HEADS * HEAD_DIM * sizeof(*reference));
    float *flash = malloc((size_t)HEADS * HEAD_DIM * sizeof(*flash));
    float sinks[HEADS];
    int *indices = malloc((size_t)topk * sizeof(*indices));
    int *window_indices = malloc((size_t)window_size * sizeof(*window_indices));
    int *compressed_indices = compressed_selected
        ? malloc((size_t)compressed_selected * sizeof(*compressed_indices)) : NULL;
    if (!kv || !queries || !reference || !flash || !indices || !window_indices ||
        (compressed_selected && !compressed_indices)) {
        free(compressed_indices); free(window_indices); free(indices);
        free(flash); free(reference); free(queries); free(kv);
        return 1;
    }
    for (size_t i = 0; i < values; i++) kv[i] = random_float();
    for (size_t i = 0; i < (size_t)HEADS * HEAD_DIM; i++)
        queries[i] = random_float();
    for (int head = 0; head < HEADS; head++) sinks[head] = -10.0f;
    for (int i = 0; i < window_size; i++) {
        window_indices[i] = padded && i > 0 && i % 11 == 0 ? -1 : i;
        indices[i] = window_indices[i];
    }
    for (int i = 0; i < compressed_selected; i++) {
        compressed_indices[i] = padded && i % 13 == 12 ? -1 : i;
        indices[window_size + i] = compressed_indices[i] < 0
            ? -1 : window_size + compressed_indices[i];
    }
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    set_flash(flash_enabled ? "1" : "0");
    int failed = coli_v4_sparse_attention_ref(
        reference, queries, kv, sinks, indices, HEADS, HEAD_DIM, topk, topk,
        scale) || coli_v4_attention_two_source_ref(
        flash, queries, kv, window_size,
        compressed_selected ? kv + (size_t)window_size * HEAD_DIM : NULL,
        compressed_selected, window_indices, compressed_indices,
        compressed_selected,
        sinks, HEADS, HEAD_DIM, scale);
    char label[64];
    snprintf(label, sizeof(label), "flash=%d topk=%d padded=%d",
             flash_enabled, topk, padded);
    size_t output_count = (size_t)HEADS * HEAD_DIM;
    if (!failed && !flash_enabled &&
        memcmp(reference, flash, output_count * sizeof(*reference))) {
        fprintf(stderr, "%s: two-pass output is not bit-identical\n", label);
        failed = 1;
    }
    if (!failed)
        failed = compare_outputs(reference, flash, output_count, label);
    free(compressed_indices); free(window_indices); free(indices);
    free(flash); free(reference); free(queries); free(kv);
    return failed;
}

static int test_random_cases(void) {
    const int topks[] = {1, 7, 64, 2048};
    for (int flash_enabled = 0; flash_enabled <= 1; flash_enabled++)
        for (size_t i = 0; i < sizeof(topks) / sizeof(topks[0]); i++) {
            if (compare_random_case(topks[i], 0, flash_enabled)) return 1;
            if (topks[i] > 1 &&
                compare_random_case(topks[i], 1, flash_enabled)) return 1;
        }
    return 0;
}

static int test_edge_cases(void) {
    float output[8], query[8] = {0}, window[8] = {2, -1, 4, 3, 1, 1, 1, 1};
    int valid[2] = {0, 1}, empty[2] = {-1, -1};
    float sinks[2] = {100.0f, -100.0f};
    if (coli_v4_flash_attention_ref(output, query, window, 2, NULL, 0,
                                    valid, NULL, 0, sinks, 2, 4, 1.0f))
        return 1;
    for (int i = 0; i < 4; i++) {
        if (output[i] != 0.0f) return 1;
        float expected = coli_bf16_round((window[i] + window[4 + i]) / 2.0f);
        if (output[4 + i] != expected) return 1;
    }
    if (coli_v4_flash_attention_ref(output, query, window, 2, NULL, 0,
                                    empty, NULL, 0, sinks, 2, 4, 1.0f) != -1)
        return 1;

    float dominant_query[2] = {1, 0};
    float dominant_values[4] = {0, -9, 100, 4};
    int dominant_indices[2] = {0, 1};
    float sink = -100.0f;
    if (coli_v4_flash_attention_ref(output, dominant_query, dominant_values, 2,
                                    NULL, 0, dominant_indices, NULL, 0, &sink,
                                    1, 2, 1.0f))
        return 1;
    if (output[0] != coli_bf16_round(100.0f) ||
        output[1] != coli_bf16_round(4.0f))
        return 1;

    int one_index = 0;
    float one_query[2] = {0, 0}, one_value[2] = {6, -2}, one_sink = 0;
    if (coli_v4_flash_attention_ref(output, one_query, one_value, 1, NULL, 0,
                                    &one_index, NULL, 0, &one_sink,
                                    1, 2, 1.0f))
        return 1;
    if (output[0] != coli_bf16_round(3.0f) ||
        output[1] != coli_bf16_round(-1.0f))
        return 1;
    return 0;
}

static void set_flash(const char *value) {
#ifdef _WIN32
    _putenv_s("V4_FLASH", value);
#else
    setenv("V4_FLASH", value, 1);
#endif
}

static int test_compressed_bounds(void) {
    float output[2], query[2] = {1, 0};
    float window[2] = {1, 0}, compressed[2] = {0, 1}, sink = 0;
    int window_index = 0, compressed_index = 1;
    for (int flash_enabled = 0; flash_enabled <= 1; flash_enabled++) {
        set_flash(flash_enabled ? "1" : "0");
        if (coli_v4_attention_two_source_ref(
                output, query, window, 1, compressed, 1, &window_index,
                &compressed_index, 1, &sink, 1, 2, 1.0f) != -1)
            return 1;
    }
    return 0;
}

static int test_switch(void) {
    float query[2] = {1, 0}, values[4] = {1, 0, 0, 1};
    float expected[2], actual[2], flash[2], sink = 0;
    int indices[2] = {0, 1};
    if (coli_v4_sparse_attention_ref(expected, query, values, &sink, indices,
                                     1, 2, 2, 2, 1.0f))
        return 1;
    set_flash("0");
    if (coli_v4_attention_two_source_ref(
            actual, query, values, 2, NULL, 0, indices, NULL, 0,
            &sink, 1, 2, 1.0f) || memcmp(expected, actual, sizeof(actual)))
        return 1;
    set_flash("garbage");
    if (coli_v4_attention_two_source_ref(
            actual, query, values, 2, NULL, 0, indices, NULL, 0,
            &sink, 1, 2, 1.0f) ||
        coli_v4_flash_attention_ref(
            flash, query, values, 2, NULL, 0, indices, NULL, 0,
            &sink, 1, 2, 1.0f) || memcmp(actual, flash, sizeof(actual)))
        return 1;
    set_flash("1");
    return 0;
}

static double monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

static int benchmark_long_context(void) {
    enum { HEADS = 64, HEAD_DIM = 512, WINDOW = 128, COMPRESSED = 32768,
           SELECTED = 512, REPEATS = 10 };
    size_t rows = WINDOW + COMPRESSED;
    float *values = malloc(rows * HEAD_DIM * sizeof(*values));
    float *staged = malloc(rows * HEAD_DIM * sizeof(*staged));
    float *queries = malloc((size_t)HEADS * HEAD_DIM * sizeof(*queries));
    float *output = malloc((size_t)HEADS * HEAD_DIM * sizeof(*output));
    int *indices = malloc((WINDOW + SELECTED) * sizeof(*indices));
    int *window_indices = malloc(WINDOW * sizeof(*window_indices));
    int *compressed_indices = malloc(SELECTED * sizeof(*compressed_indices));
    float sinks[HEADS];
    if (!values || !staged || !queries || !output || !indices ||
        !window_indices || !compressed_indices) {
        free(compressed_indices); free(window_indices); free(indices);
        free(output); free(queries); free(staged); free(values);
        return 1;
    }
    for (size_t i = 0; i < rows * HEAD_DIM; i++) values[i] = random_float();
    for (size_t i = 0; i < (size_t)HEADS * HEAD_DIM; i++)
        queries[i] = random_float();
    for (int head = 0; head < HEADS; head++) sinks[head] = random_float();
    for (int i = 0; i < WINDOW; i++) indices[i] = window_indices[i] = i;
    for (int i = 0; i < SELECTED; i++) {
        compressed_indices[i] = i * 63;
        indices[WINDOW + i] = WINDOW + compressed_indices[i];
    }
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    double start = monotonic_seconds();
    for (int repeat = 0; repeat < REPEATS; repeat++) {
        memcpy(staged, values, rows * HEAD_DIM * sizeof(*staged));
        if (coli_v4_sparse_attention_ref(output, queries, staged, sinks, indices,
                                         HEADS, HEAD_DIM, (int)rows,
                                         WINDOW + SELECTED, scale)) return 1;
    }
    double legacy = monotonic_seconds() - start;
    start = monotonic_seconds();
    for (int repeat = 0; repeat < REPEATS; repeat++)
        if (coli_v4_flash_attention_ref(
                output, queries, values, WINDOW,
                values + (size_t)WINDOW * HEAD_DIM,
                COMPRESSED, window_indices, compressed_indices, SELECTED,
                sinks, HEADS, HEAD_DIM, scale)) return 1;
    double flash = monotonic_seconds() - start;
    double legacy_ms = legacy * 1000.0 / REPEATS;
    double flash_ms = flash * 1000.0 / REPEATS;
    printf("128k CSA attention: legacy %.3f ms/layer, flash %.3f ms/layer, %.2fx; "
           "21-layer contribution %.3f -> %.3f ms/token\n",
           legacy_ms, flash_ms, legacy / flash,
           legacy_ms * 21.0, flash_ms * 21.0);
    free(compressed_indices); free(window_indices); free(indices);
    free(output); free(queries); free(staged); free(values);
    return 0;
}

int main(int argc, char **argv) {
    if (test_random_cases() || test_edge_cases() || test_compressed_bounds() ||
        test_switch()) {
        fprintf(stderr, "test_v4_flash_attention: FAIL\n");
        return 1;
    }
    if (argc == 2 && strcmp(argv[1], "--benchmark") == 0 &&
        benchmark_long_context()) {
        fprintf(stderr, "test_v4_flash_attention benchmark: FAIL\n");
        return 1;
    }
    puts("test_v4_flash_attention: ok");
    return 0;
}
