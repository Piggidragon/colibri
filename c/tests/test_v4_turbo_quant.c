#include "../native_quant.h"
#include "../turbo_quant.h"
#include "../v4_kv_codec.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t random_state = UINT64_C(0x9e3779b97f4a7c15);

static float uniform_open(void) {
    random_state = random_state * UINT64_C(6364136223846793005) +
                   UINT64_C(1442695040888963407);
    return ((float)((random_state >> 40) + 1)) / 16777217.0f;
}

static float normal_random(void) {
    float radius = sqrtf(-2.0f * logf(uniform_open()));
    return radius * cosf(6.2831853071795864769f * uniform_open());
}

static double dot(const float *left, const float *right, int length) {
    double result = 0.0;
    for (int i = 0; i < length; i++) result += (double)left[i] * right[i];
    return result;
}

static int test_wht(void) {
    float left[COLI_TQ_GROUP], right[COLI_TQ_GROUP];
    float left_rotated[COLI_TQ_GROUP], right_rotated[COLI_TQ_GROUP];
    for (int i = 0; i < COLI_TQ_GROUP; i++) {
        left[i] = normal_random();
        right[i] = normal_random();
    }
    memcpy(left_rotated, left, sizeof(left));
    memcpy(right_rotated, right, sizeof(right));
    double norm_before = dot(left, left, COLI_TQ_GROUP);
    double product_before = dot(left, right, COLI_TQ_GROUP);
    coli_tq_fwht(left_rotated);
    coli_tq_fwht(right_rotated);
    double norm_after = dot(left_rotated, left_rotated, COLI_TQ_GROUP);
    double product_after = dot(left_rotated, right_rotated, COLI_TQ_GROUP);
    if (fabs(norm_after - norm_before) > 1e-5 * norm_before ||
        fabs(product_after - product_before) >
            1e-5 * fmax(1.0, fabs(product_before))) {
        fprintf(stderr, "WHT orthogonality failed\n");
        return 1;
    }
    coli_tq_fwht_inverse(left_rotated);
    for (int i = 0; i < COLI_TQ_GROUP; i++) {
        if (fabsf(left_rotated[i] - left[i]) >
            1e-6f * fmaxf(1.0f, fabsf(left[i]))) {
            fprintf(stderr, "WHT inverse failed at %d\n", i);
            return 1;
        }
    }
    return 0;
}

static int test_fp16(void) {
    static const struct { float value; uint16_t bits; } exact[] = {
        {0.0f, 0x0000}, {-0.0f, 0x8000}, {1.0f, 0x3c00},
        {-2.0f, 0xc000}, {65504.0f, 0x7bff}, {0x1p-24f, 0x0001},
    };
    for (size_t i = 0; i < sizeof(exact) / sizeof(exact[0]); i++) {
        uint16_t encoded = coli_tq_fp16_encode(exact[i].value);
        float decoded = coli_tq_fp16_decode(encoded);
        if (encoded != exact[i].bits ||
            memcmp(&decoded, &exact[i].value, sizeof(decoded))) {
            fprintf(stderr, "fp16 exact case %zu failed\n", i);
            return 1;
        }
    }
    float values[] = {0x1p-20f, -0x1p-20f, 0.001f, -17.25f, 4095.0f};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
        float decoded = coli_tq_fp16_decode(coli_tq_fp16_encode(values[i]));
        if (!isfinite(decoded) || fabsf(decoded - values[i]) >
            fabsf(values[i]) * 0.001f) {
            fprintf(stderr, "fp16 round-trip case %zu failed\n", i);
            return 1;
        }
    }
    if (!isinf(coli_tq_fp16_decode(coli_tq_fp16_encode(INFINITY))) ||
        !isnan(coli_tq_fp16_decode(coli_tq_fp16_encode(NAN)))) {
        fprintf(stderr, "fp16 non-finite handling failed\n");
        return 1;
    }
    return 0;
}

static int reference_nearest3(float value) {
    static const float centroids[8] = {
        -0.190207f, -0.118786f, -0.066822f, -0.021663f,
         0.021663f,  0.066822f,  0.118786f,  0.190207f,
    };
    int best = 0;
    float distance = fabsf(value - centroids[0]);
    for (int i = 1; i < 8; i++) {
        float candidate = fabsf(value - centroids[i]);
        if (candidate < distance) {
            best = i;
            distance = candidate;
        }
    }
    return best;
}

static int test_turbo3_packing(void) {
    float source[COLI_TQ_GROUP], rotated[COLI_TQ_GROUP];
    for (int i = 0; i < COLI_TQ_GROUP; i++) source[i] = normal_random();
    double norm_squared = dot(source, source, COLI_TQ_GROUP);
    float inverse_norm = 1.0f / sqrtf((float)norm_squared);
    for (int i = 0; i < COLI_TQ_GROUP; i++) rotated[i] = source[i] * inverse_norm;
    coli_tq_fwht(rotated);

    ColiTurbo3Block block;
    uint8_t expected_low[32] = {0}, expected_high[16] = {0};
    if (coli_tq_encode_group(&block, source, 3)) return 1;
    for (int i = 0; i < COLI_TQ_GROUP; i++) {
        int index = reference_nearest3(rotated[i]);
        expected_low[i / 4] |= (uint8_t)((index & 3) << ((i % 4) * 2));
        expected_high[i / 8] |= (uint8_t)((index >> 2) << (i % 8));
    }
    if (memcmp(block.qs, expected_low, sizeof(expected_low)) ||
        memcmp(block.signs, expected_high, sizeof(expected_high))) {
        fprintf(stderr, "turbo3 packing failed\n");
        return 1;
    }
    return 0;
}

typedef struct {
    double cosine;
    double norm_ratio;
    double nope_cosine;
    double rope_cosine;
} Quality;

static int measure_quality(ColiV4KVCodec codec, int fp8_input,
                           float rope_scale,
                           Quality *quality) {
    enum { DIMENSION = 512, ROWS = 32, NOPE = 448 };
    size_t row_bytes = coli_v4_kv_row_bytes(
        codec, COLI_V4_KV_MAIN, DIMENSION, 64);
    unsigned char *encoded = malloc(row_bytes);
    float source[DIMENSION], prepared[DIMENSION], decoded[DIMENSION];
    uint8_t scales[(NOPE + 63) / 64];
    double source_squared = 0.0, decoded_squared = 0.0, product = 0.0;
    double nope_source_squared = 0.0, nope_decoded_squared = 0.0;
    double nope_product = 0.0, rope_source_squared = 0.0;
    double rope_decoded_squared = 0.0, rope_product = 0.0;
    if (!encoded) return 1;
    /* Continuous and prequantized runs see the same underlying vectors. */
    random_state = UINT64_C(0x4d595df4d0f33173);
    for (int row = 0; row < ROWS; row++) {
        for (int i = 0; i < DIMENSION; i++) source[i] = normal_random();
        for (int i = NOPE; i < DIMENSION; i++) source[i] *= rope_scale;
        if (fp8_input) {
            if (coli_fp8_activation_qdq_ref(
                    prepared, scales, source, NOPE, 64)) {
                free(encoded);
                return 1;
            }
            coli_bf16_round_array(prepared, NOPE);
            for (int i = NOPE; i < DIMENSION; i++)
                prepared[i] = coli_bf16_round(source[i]);
        } else {
            memcpy(prepared, source, sizeof(prepared));
        }
        if (coli_v4_kv_encode_row(
                codec, COLI_V4_KV_MAIN, encoded, prepared, DIMENSION, 64) ||
            coli_v4_kv_decode_row(
                codec, COLI_V4_KV_MAIN, decoded, encoded, DIMENSION, 64)) {
            free(encoded);
            return 1;
        }
        source_squared += dot(prepared, prepared, DIMENSION);
        decoded_squared += dot(decoded, decoded, DIMENSION);
        product += dot(prepared, decoded, DIMENSION);
        nope_source_squared += dot(prepared, prepared, NOPE);
        nope_decoded_squared += dot(decoded, decoded, NOPE);
        nope_product += dot(prepared, decoded, NOPE);
        rope_source_squared += dot(
            prepared + NOPE, prepared + NOPE, DIMENSION - NOPE);
        rope_decoded_squared += dot(
            decoded + NOPE, decoded + NOPE, DIMENSION - NOPE);
        rope_product += dot(
            prepared + NOPE, decoded + NOPE, DIMENSION - NOPE);
    }
    free(encoded);
    quality->cosine = product / sqrt(source_squared * decoded_squared);
    quality->norm_ratio = sqrt(decoded_squared / source_squared);
    quality->nope_cosine = nope_product /
        sqrt(nope_source_squared * nope_decoded_squared);
    quality->rope_cosine = rope_product /
        sqrt(rope_source_squared * rope_decoded_squared);
    return 0;
}

static int test_quality(void) {
    static const struct {
        ColiV4KVCodec codec;
        double minimum_cosine;
    } codecs[] = {
        {COLI_V4_KV_TURBO2, 0.93},
        {COLI_V4_KV_TURBO3, 0.98},
        {COLI_V4_KV_TURBO4, 0.994},
    };
    for (size_t i = 0; i < sizeof(codecs) / sizeof(codecs[0]); i++) {
        Quality continuous, fp8;
        if (measure_quality(codecs[i].codec, 0, 1.0f, &continuous) ||
            measure_quality(codecs[i].codec, 1, 1.0f, &fp8))
            return 1;
        printf("%s continuous cosine=%.6f norm=%.6f; "
               "fp8-grid cosine=%.6f norm=%.6f nope=%.6f rope=%.6f\n",
               coli_v4_kv_codec_name(codecs[i].codec),
               continuous.cosine, continuous.norm_ratio,
               fp8.cosine, fp8.norm_ratio, fp8.nope_cosine, fp8.rope_cosine);
        if (continuous.cosine < codecs[i].minimum_cosine ||
            fp8.cosine < codecs[i].minimum_cosine ||
            fp8.nope_cosine < codecs[i].minimum_cosine ||
            fp8.rope_cosine < codecs[i].minimum_cosine ||
            continuous.norm_ratio < 0.98 || continuous.norm_ratio > 1.02 ||
            fp8.norm_ratio < 0.98 || fp8.norm_ratio > 1.02) {
            fprintf(stderr, "%s quality bound failed\n",
                    coli_v4_kv_codec_name(codecs[i].codec));
            return 1;
        }
    }
    return 0;
}

static int test_rope_scale_sensitivity(void) {
    static const float scales[] = {0.25f, 0.0625f, 16.0f};
    Quality quality[sizeof(scales) / sizeof(scales[0])];
    for (size_t i = 0; i < sizeof(scales) / sizeof(scales[0]); i++) {
        if (measure_quality(
                COLI_V4_KV_TURBO3, 1, scales[i], &quality[i]))
            return 1;
        printf("turbo3 rope-scale=%g cosine=%.6f norm=%.6f "
               "nope=%.6f rope=%.6f\n",
               scales[i], quality[i].cosine, quality[i].norm_ratio,
               quality[i].nope_cosine, quality[i].rope_cosine);
        if (!isfinite(quality[i].cosine) ||
            !isfinite(quality[i].nope_cosine) ||
            !isfinite(quality[i].rope_cosine) || quality[i].cosine < 0.98) {
            fprintf(stderr, "turbo3 rope-scale=%g quality failed\n", scales[i]);
            return 1;
        }
    }
    /* The aggregate cosine hides whichever 64/448-dimensional half has the
     * smaller variance.  Keep that limitation visible in this test instead of
     * treating the equal-scale result as evidence for a uniform layout. */
    if (!(quality[1].rope_cosine < 0.8 && quality[1].nope_cosine > 0.95 &&
          quality[2].nope_cosine < 0.95 && quality[2].rope_cosine > 0.95)) {
        fprintf(stderr, "TurboQuant split-scale sensitivity disappeared\n");
        return 1;
    }
    return 0;
}

static int test_rejected_norms_and_alignment(void) {
    float source[COLI_TQ_GROUP], decoded[COLI_TQ_GROUP];
    unsigned char storage[sizeof(ColiTurbo4Block) + 1];
    void *misaligned = storage + 1;
    for (int bits = 2; bits <= 4; bits++) {
        for (int i = 0; i < COLI_TQ_GROUP; i++) source[i] = 100000.0f;
        if (coli_tq_encode_group(misaligned, source, bits) != -1) {
            fprintf(stderr, "turbo%d accepted an fp16-overflowing norm\n", bits);
            return 1;
        }
        source[0] = NAN;
        if (coli_tq_encode_group(misaligned, source, bits) != -1) {
            fprintf(stderr, "turbo%d accepted a non-finite input\n", bits);
            return 1;
        }
        for (int i = 0; i < COLI_TQ_GROUP; i++)
            source[i] = (float)(i - 64) / 128.0f;
        if (coli_tq_encode_group(misaligned, source, bits) ||
            coli_tq_decode_group(decoded, misaligned, bits)) {
            fprintf(stderr, "turbo%d rejected a misaligned byte buffer\n", bits);
            return 1;
        }
        for (int i = 0; i < COLI_TQ_GROUP; i++)
            if (!isfinite(decoded[i])) return 1;
    }
    return 0;
}

static int test_codec_geometry(void) {
    if (coli_v4_kv_row_bytes(
            COLI_V4_KV_TURBO2, COLI_V4_KV_MAIN, 512, 64) != 136 ||
        coli_v4_kv_row_bytes(
            COLI_V4_KV_TURBO3, COLI_V4_KV_MAIN, 512, 64) != 200 ||
        coli_v4_kv_row_bytes(
            COLI_V4_KV_TURBO4, COLI_V4_KV_MAIN, 512, 64) != 264 ||
        coli_v4_kv_row_bytes(
            COLI_V4_KV_TURBO3, COLI_V4_KV_INDEX, 128, 0) != 50 ||
        coli_v4_kv_row_bytes(
            COLI_V4_KV_TURBO3, COLI_V4_KV_MAIN, 32, 16) != 0) {
        fprintf(stderr, "TurboQuant geometry failed\n");
        return 1;
    }
    setenv("COLI_TEST_V4_KV", "turbo3", 1);
    ColiV4KVCodec fallback = coli_v4_kv_codec_from_env(
        "COLI_TEST_V4_KV", COLI_V4_KV_MAIN, 32, 16, COLI_V4_KV_F32);
    unsetenv("COLI_TEST_V4_KV");
    if (fallback != COLI_V4_KV_F32) {
        fprintf(stderr, "TurboQuant fallback failed\n");
        return 1;
    }
    return 0;
}

static int test_zero_rows(void) {
    float source[512] = {0}, decoded[512];
    unsigned char encoded[264];
    for (ColiV4KVCodec codec = COLI_V4_KV_TURBO4;
         codec <= COLI_V4_KV_TURBO2; codec++) {
        if (coli_v4_kv_encode_row(
                codec, COLI_V4_KV_MAIN, encoded, source, 512, 64) ||
            coli_v4_kv_decode_row(
                codec, COLI_V4_KV_MAIN, decoded, encoded, 512, 64))
            return 1;
        for (int i = 0; i < 512; i++) {
            if (!isfinite(decoded[i]) || decoded[i] != 0.0f) {
                fprintf(stderr, "%s zero row failed at %d\n",
                        coli_v4_kv_codec_name(codec), i);
                return 1;
            }
        }
    }
    return 0;
}

int main(void) {
    if (test_wht() || test_fp16() || test_turbo3_packing() ||
        test_codec_geometry() || test_zero_rows() ||
        test_rejected_norms_and_alignment() || test_quality() ||
        test_rope_scale_sensitivity())
        return 1;
    puts("V4 TurboQuant tests passed");
    return 0;
}
