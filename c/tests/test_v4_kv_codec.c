#include "../v4_kv_codec.h"
#include "../native_quant.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int test_row_sizes(void) {
    static const struct {
        int dimension, rope_dim;
        size_t main_native, index_native;
    } cases[] = {
        {32, 16, 49, 17},
        {128, 64, 193, 68},
        {512, 64, 583, 272},
        {4096, 64, 4223, 2176},
    };
    for (size_t item = 0; item < sizeof(cases) / sizeof(cases[0]); item++) {
        int dimension = cases[item].dimension;
        if (coli_v4_kv_row_bytes(
                COLI_V4_KV_F32, COLI_V4_KV_MAIN, dimension,
                cases[item].rope_dim) != (size_t)dimension * sizeof(float) ||
            coli_v4_kv_row_bytes(
                COLI_V4_KV_F32, COLI_V4_KV_INDEX, dimension, 0) !=
                (size_t)dimension * sizeof(float) ||
            coli_v4_kv_row_bytes(
                COLI_V4_KV_NATIVE, COLI_V4_KV_MAIN, dimension,
                cases[item].rope_dim) != cases[item].main_native ||
            coli_v4_kv_row_bytes(
                COLI_V4_KV_NATIVE, COLI_V4_KV_INDEX, dimension, 0) !=
                cases[item].index_native)
            return 1;
        static const size_t block_bytes[] = {66, 50, 34};
        for (ColiV4KVCodec codec = COLI_V4_KV_TURBO4;
             codec <= COLI_V4_KV_TURBO2; codec++) {
            size_t expected = dimension % 128 ? 0
                : (size_t)(dimension / 128) *
                  block_bytes[codec - COLI_V4_KV_TURBO4];
            if (coli_v4_kv_row_bytes(codec, COLI_V4_KV_MAIN, dimension,
                                     cases[item].rope_dim) != expected ||
                coli_v4_kv_row_bytes(codec, COLI_V4_KV_INDEX,
                                     dimension, 0) != expected)
                return 1;
        }
    }
    return 0;
}

static int check_native_row(ColiV4KVStream stream, float *legacy,
                            int dimension, int rope_dim) {
    size_t bytes = coli_v4_kv_row_bytes(
        COLI_V4_KV_NATIVE, stream, dimension, rope_dim);
    unsigned char *encoded = malloc(bytes);
    float *decoded = malloc((size_t)dimension * sizeof(*decoded));
    if (!encoded || !decoded) {
        free(decoded); free(encoded); return 1;
    }
    int failed = coli_v4_kv_encode_row(
        COLI_V4_KV_NATIVE, stream, encoded, legacy, dimension, rope_dim);
    if (failed) fprintf(stderr, "encode failed stream=%d\n", stream);
    if (!failed) failed = coli_v4_kv_decode_row(
        COLI_V4_KV_NATIVE, stream, decoded, encoded, dimension, rope_dim);
    if (!failed && memcmp(legacy, decoded,
                          (size_t)dimension * sizeof(*legacy))) {
        fprintf(stderr, "decode mismatch stream=%d\n", stream);
        failed = 1;
    }
    free(decoded); free(encoded);
    return failed;
}

static int test_native_main(void) {
    enum { DIMENSION = 512, ROPE = 64, NOPE = DIMENSION - ROPE };
    float raw[DIMENSION], legacy[DIMENSION];
    uint8_t scales[(NOPE + 63) / 64];
    for (int i = 0; i < DIMENSION; i++)
        raw[i] = (float)((i * 37) % 257 - 128) / 9.0f;
    raw[0] = 0.0f;
    raw[1] = 448.0f;
    raw[2] = -448.0f;
    raw[3] = FLT_MAX;
    raw[4] = -FLT_MAX;
    if (coli_fp8_activation_qdq_ref(legacy, scales, raw, NOPE, 64)) return 1;
    coli_bf16_round_array(legacy, NOPE);
    for (int i = NOPE; i < DIMENSION; i++) legacy[i] = coli_bf16_round(raw[i]);
    if (check_native_row(COLI_V4_KV_MAIN, legacy, DIMENSION, ROPE)) return 1;
    memset(raw, 0, sizeof(raw));
    if (coli_fp8_activation_qdq_ref(legacy, scales, raw, NOPE, 64)) return 1;
    coli_bf16_round_array(legacy, NOPE);
    memset(legacy + NOPE, 0, ROPE * sizeof(*legacy));
    return check_native_row(COLI_V4_KV_MAIN, legacy, DIMENSION, ROPE);
}

static int test_native_main_rope_edges(void) {
    enum { DIMENSION = 64 };
    float raw[DIMENSION], legacy[DIMENSION];
    uint8_t scales[1];
    for (int i = 0; i < DIMENSION; i++)
        raw[i] = (float)((i * 17) % 61 - 30) / 5.0f;

    memcpy(legacy, raw, sizeof(legacy));
    coli_bf16_round_array(legacy, DIMENSION);
    if (check_native_row(
            COLI_V4_KV_MAIN, legacy, DIMENSION, DIMENSION)) return 1;

    if (coli_fp8_activation_qdq_ref(
            legacy, scales, raw, DIMENSION, DIMENSION)) return 1;
    coli_bf16_round_array(legacy, DIMENSION);
    return check_native_row(COLI_V4_KV_MAIN, legacy, DIMENSION, 0);
}

static int test_native_index(void) {
    enum { DIMENSION = 128 };
    float legacy[DIMENSION], qdq[DIMENSION];
    uint8_t scales[(DIMENSION + 31) / 32];
    for (int i = 0; i < DIMENSION; i++)
        legacy[i] = (float)((i * 23) % 97 - 48) / 7.0f;
    legacy[0] = FLT_MAX;
    legacy[1] = -FLT_MAX;
    if (coli_hadamard_bf16_ref(legacy, DIMENSION) ||
        coli_fp4_activation_qdq_ref(qdq, scales, legacy, DIMENSION, 32))
        return 1;
    memcpy(legacy, qdq, sizeof(legacy));
    coli_bf16_round_array(legacy, DIMENSION);
    if (check_native_row(COLI_V4_KV_INDEX, legacy, DIMENSION, 0)) return 1;
    memset(legacy, 0, sizeof(legacy));
    if (coli_fp4_activation_qdq_ref(qdq, scales, legacy, DIMENSION, 32)) return 1;
    memcpy(legacy, qdq, sizeof(legacy));
    coli_bf16_round_array(legacy, DIMENSION);
    return check_native_row(COLI_V4_KV_INDEX, legacy, DIMENSION, 0);
}

static int test_native_rejection_and_signed_zero(void) {
    enum { DIMENSION = 128, ROPE = 64 };
    float invalid[DIMENSION] = {0};
    unsigned char main_row[193], index_row[68], decoded_bytes[68];
    float decoded[DIMENSION];
    invalid[0] = nextafterf(1.0f, 2.0f);
    if (!coli_v4_kv_encode_row(
            COLI_V4_KV_NATIVE, COLI_V4_KV_MAIN,
            main_row, invalid, DIMENSION, ROPE) ||
        !coli_v4_kv_encode_row(
            COLI_V4_KV_NATIVE, COLI_V4_KV_INDEX,
            index_row, invalid, DIMENSION, 0))
        return 1;

    memset(invalid, 0, sizeof(invalid));
    invalid[0] = -0.0f;
    if (coli_v4_kv_encode_row(
            COLI_V4_KV_NATIVE, COLI_V4_KV_INDEX,
            decoded_bytes, invalid, DIMENSION, 0) ||
        (decoded_bytes[0] & 0x0f) != 8 ||
        coli_v4_kv_decode_row(
            COLI_V4_KV_NATIVE, COLI_V4_KV_INDEX,
            decoded, decoded_bytes, DIMENSION, 0) ||
        !signbit(decoded[0]))
        return 1;
    return 0;
}

static int test_f32_roundtrip(void) {
    enum { DIMENSION = 128 };
    float input[DIMENSION], encoded[DIMENSION], decoded[DIMENSION];
    for (int i = 0; i < DIMENSION; i++)
        input[i] = (float)(i - 63) / 17.0f;
    if (coli_v4_kv_encode_row(COLI_V4_KV_F32, COLI_V4_KV_MAIN,
                              encoded, input, DIMENSION, 32) ||
        coli_v4_kv_decode_row(COLI_V4_KV_F32, COLI_V4_KV_MAIN,
                              decoded, encoded, DIMENSION, 32) ||
        memcmp(input, decoded, sizeof(input)))
        return 1;
    return 0;
}

static int test_names_and_env(void) {
    if (strcmp(coli_v4_kv_codec_name(COLI_V4_KV_F32), "f32") ||
        strcmp(coli_v4_kv_codec_name(COLI_V4_KV_NATIVE), "native") ||
        strcmp(coli_v4_kv_codec_name(COLI_V4_KV_TURBO4), "turbo4") ||
        strcmp(coli_v4_kv_codec_name(COLI_V4_KV_TURBO3), "turbo3") ||
        strcmp(coli_v4_kv_codec_name(COLI_V4_KV_TURBO2), "turbo2"))
        return 1;
    setenv("COLI_TEST_V4_KV", "f32", 1);
    if (coli_v4_kv_codec_from_env(
            "COLI_TEST_V4_KV", COLI_V4_KV_MAIN, 512, 64,
            COLI_V4_KV_F32) != COLI_V4_KV_F32)
        return 1;
    setenv("COLI_TEST_V4_KV", "native", 1);
    if (coli_v4_kv_codec_from_env(
            "COLI_TEST_V4_KV", COLI_V4_KV_MAIN, 512, 64,
            COLI_V4_KV_F32) != COLI_V4_KV_NATIVE)
        return 1;
    setenv("COLI_TEST_V4_KV", "garbage", 1);
    if (coli_v4_kv_codec_from_env(
            "COLI_TEST_V4_KV", COLI_V4_KV_MAIN, 512, 64,
            COLI_V4_KV_F32) != COLI_V4_KV_F32)
        return 1;
    if (coli_v4_kv_codec_from_env(
            "COLI_TEST_V4_KV", COLI_V4_KV_MAIN, 512, 64,
            COLI_V4_KV_TURBO3) != COLI_V4_KV_TURBO3)
        return 1;
    setenv("COLI_TEST_V4_KV", "turbo3", 1);
    if (coli_v4_kv_codec_from_env(
            "COLI_TEST_V4_KV", COLI_V4_KV_MAIN, 512, 64,
            COLI_V4_KV_NATIVE) != COLI_V4_KV_TURBO3)
        return 1;
    if (coli_v4_kv_codec_from_env(
            "COLI_TEST_V4_KV", COLI_V4_KV_MAIN, 32, 16,
            COLI_V4_KV_NATIVE) != COLI_V4_KV_NATIVE)
        return 1;
    unsetenv("COLI_TEST_V4_KV");
    return 0;
}

int main(void) {
    if (test_row_sizes()) { fprintf(stderr, "row sizes failed\n"); return 1; }
    if (test_f32_roundtrip()) { fprintf(stderr, "f32 failed\n"); return 1; }
    if (test_native_main()) { fprintf(stderr, "native main failed\n"); return 1; }
    if (test_native_main_rope_edges()) {
        fprintf(stderr, "native main RoPE edges failed\n"); return 1;
    }
    if (test_native_index()) { fprintf(stderr, "native index failed\n"); return 1; }
    if (test_native_rejection_and_signed_zero()) {
        fprintf(stderr, "native rejection/signed zero failed\n"); return 1;
    }
    if (test_names_and_env()) { fprintf(stderr, "environment failed\n"); return 1; }
    puts("V4 KV codec tests passed");
    return 0;
}
