#include "../v4_kv_codec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int test_f32_rows(void) {
    const int dimensions[] = {32, 128, 512, 4096};
    for (size_t i = 0; i < sizeof(dimensions) / sizeof(dimensions[0]); i++) {
        int dimension = dimensions[i];
        if (coli_v4_kv_row_bytes(COLI_V4_KV_F32, COLI_V4_KV_MAIN,
                                 dimension, dimension / 2) !=
            (size_t)dimension * sizeof(float))
            return 1;
        if (coli_v4_kv_row_bytes(COLI_V4_KV_F32, COLI_V4_KV_INDEX,
                                 dimension, 0) !=
            (size_t)dimension * sizeof(float))
            return 1;
    }
    if (coli_v4_kv_row_bytes(COLI_V4_KV_NATIVE, COLI_V4_KV_MAIN,
                             512, 64) != 0 ||
        coli_v4_kv_row_bytes(COLI_V4_KV_TURBO3, COLI_V4_KV_INDEX,
                             128, 0) != 0)
        return 1;
    return 0;
}

static int test_f32_operations(void) {
    enum { DIMENSION = 128 };
    float input[DIMENSION], encoded[DIMENSION], decoded[DIMENSION];
    float query[DIMENSION], actual[DIMENSION], expected[DIMENSION];
    for (int i = 0; i < DIMENSION; i++) {
        input[i] = (float)(i - 63) / 17.0f;
        query[i] = (float)((i * 13) % 29 - 14) / 11.0f;
        actual[i] = expected[i] = (float)(i % 7) / 9.0f;
    }
    if (coli_v4_kv_encode_row(COLI_V4_KV_F32, COLI_V4_KV_MAIN,
                              encoded, input, DIMENSION, 32) ||
        coli_v4_kv_decode_row(COLI_V4_KV_F32, COLI_V4_KV_MAIN,
                              decoded, encoded, DIMENSION, 32) ||
        memcmp(input, decoded, sizeof(input)))
        return 1;
    float expected_dot = 0.0f;
    for (int i = 0; i < DIMENSION; i++) expected_dot += query[i] * input[i];
    float actual_dot = coli_v4_kv_dot(
        COLI_V4_KV_F32, COLI_V4_KV_MAIN, query, encoded, DIMENSION, 32);
    if (actual_dot != expected_dot) return 1;
    const float probability = 0.375f;
    for (int i = 0; i < DIMENSION; i++)
        expected[i] += probability * input[i];
    coli_v4_kv_accumulate(COLI_V4_KV_F32, COLI_V4_KV_MAIN,
                          actual, probability, encoded, DIMENSION, 32);
    return memcmp(actual, expected, sizeof(actual)) != 0;
}

static int test_names_and_env(void) {
    if (strcmp(coli_v4_kv_codec_name(COLI_V4_KV_F32), "f32") ||
        strcmp(coli_v4_kv_codec_name(COLI_V4_KV_NATIVE), "native"))
        return 1;
    setenv("COLI_TEST_V4_KV", "f32", 1);
    if (coli_v4_kv_codec_from_env(
            "COLI_TEST_V4_KV", COLI_V4_KV_MAIN, 512, 64,
            COLI_V4_KV_F32) != COLI_V4_KV_F32)
        return 1;
    setenv("COLI_TEST_V4_KV", "garbage", 1);
    if (coli_v4_kv_codec_from_env(
            "COLI_TEST_V4_KV", COLI_V4_KV_MAIN, 512, 64,
            COLI_V4_KV_F32) != COLI_V4_KV_F32)
        return 1;
    unsetenv("COLI_TEST_V4_KV");
    return 0;
}

int main(void) {
    if (test_f32_rows() || test_f32_operations() || test_names_and_env()) {
        fprintf(stderr, "V4 KV codec test failed\n");
        return 1;
    }
    puts("V4 KV codec tests passed");
    return 0;
}
