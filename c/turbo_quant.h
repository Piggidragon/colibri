#ifndef COLIBRI_TURBO_QUANT_H
#define COLIBRI_TURBO_QUANT_H

/* TurboQuant PolarQuant codec for 128-value KV groups.
 *
 * Derived from llama.cpp-turboquant's ggml-turbo-quant.c (MIT), based on
 * TurboQuant, arXiv:2504.19874.  Portions copyright (c) 2023-2026 The ggml
 * authors, used under the MIT License.  The unused QJL/matrix path is
 * intentionally omitted: the reference's default 4-bit format is plain
 * PolarQuant.
 *
 * The centroid tables are calibrated for N(0, 1/sqrt(128)).  Changing the WHT
 * group size requires rescaling the tables, so the public helpers accept only
 * exact 128-value groups.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_TQ_GROUP 128

typedef struct {
    uint16_t norm;
    uint8_t qs[32];
} ColiTurbo2Block;

typedef struct {
    uint16_t norm;
    uint8_t qs[32];
    uint8_t signs[16];
} ColiTurbo3Block;

typedef struct {
    uint16_t norm;
    uint8_t qs[64];
} ColiTurbo4Block;

#if defined(__cplusplus)
static_assert(sizeof(ColiTurbo2Block) == 34, "invalid Turbo2 block layout");
static_assert(sizeof(ColiTurbo3Block) == 50, "invalid Turbo3 block layout");
static_assert(sizeof(ColiTurbo4Block) == 66, "invalid Turbo4 block layout");
#else
_Static_assert(sizeof(ColiTurbo2Block) == 34, "invalid Turbo2 block layout");
_Static_assert(sizeof(ColiTurbo3Block) == 50, "invalid Turbo3 block layout");
_Static_assert(sizeof(ColiTurbo4Block) == 66, "invalid Turbo4 block layout");
#endif

static const float coli_tq_centroids2[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f,
};

static const float coli_tq_centroids3[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f,
};

static const float coli_tq_centroids4[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f,
};

/* Seed-42 sign tables from the reference implementation. */
static const float coli_tq_s1[COLI_TQ_GROUP] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,
};

static const float coli_tq_s2[COLI_TQ_GROUP] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,
};

/* IEEE-754 binary16 conversion, round-to-nearest-even. */
static inline uint16_t coli_tq_fp16_encode(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & UINT32_C(0x8000);
    uint32_t exponent = (bits >> 23) & UINT32_C(0xff);
    uint32_t mantissa = bits & UINT32_C(0x7fffff);
    if (exponent == 0xff)
        return (uint16_t)(sign | UINT32_C(0x7c00) |
                          (mantissa ? UINT32_C(0x0200) | (mantissa >> 13) : 0));
    int32_t half_exponent = (int32_t)exponent - 127 + 15;
    if (half_exponent >= 0x1f) return (uint16_t)(sign | UINT32_C(0x7c00));
    if (half_exponent <= 0) {
        if (half_exponent < -10) return (uint16_t)sign;
        mantissa |= UINT32_C(0x800000);
        uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1);
        uint32_t halfway = UINT32_C(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1)))
            rounded++;
        return (uint16_t)(sign | rounded);
    }
    uint32_t rounded = mantissa >> 13;
    uint32_t remainder = mantissa & UINT32_C(0x1fff);
    uint32_t output = ((uint32_t)half_exponent << 10) | rounded;
    if (remainder > UINT32_C(0x1000) ||
        (remainder == UINT32_C(0x1000) && (rounded & 1)))
        output++;
    return (uint16_t)(sign | output);
}

static inline float coli_tq_fp16_decode(uint16_t value) {
    uint32_t sign = (uint32_t)(value >> 15) << 31;
    uint32_t exponent = (value >> 10) & UINT32_C(0x1f);
    uint32_t mantissa = value & UINT32_C(0x3ff);
    uint32_t bits;
    if (!exponent) {
        if (!mantissa) bits = sign;
        else {
            int shift = 0;
            while (!(mantissa & UINT32_C(0x400))) {
                mantissa <<= 1;
                shift++;
            }
            bits = sign | ((uint32_t)(127 - 14 - shift) << 23) |
                   ((mantissa & UINT32_C(0x3ff)) << 13);
        }
    } else if (exponent == UINT32_C(0x1f)) {
        bits = sign | UINT32_C(0x7f800000) | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static inline void coli_tq_fwht(float values[COLI_TQ_GROUP]) {
    for (int i = 0; i < COLI_TQ_GROUP; i++) values[i] *= coli_tq_s1[i];
    for (int width = 1; width < COLI_TQ_GROUP; width *= 2) {
        for (int base = 0; base < COLI_TQ_GROUP; base += width * 2) {
            for (int i = base; i < base + width; i++) {
                float left = values[i], right = values[i + width];
                values[i] = left + right;
                values[i + width] = left - right;
            }
        }
    }
    for (int i = 0; i < COLI_TQ_GROUP; i++)
        values[i] *= 0.08838834764831845f * coli_tq_s2[i];
}

static inline void coli_tq_fwht_inverse(float values[COLI_TQ_GROUP]) {
    for (int i = 0; i < COLI_TQ_GROUP; i++) values[i] *= coli_tq_s2[i];
    for (int width = 1; width < COLI_TQ_GROUP; width *= 2) {
        for (int base = 0; base < COLI_TQ_GROUP; base += width * 2) {
            for (int i = base; i < base + width; i++) {
                float left = values[i], right = values[i + width];
                values[i] = left + right;
                values[i + width] = left - right;
            }
        }
    }
    for (int i = 0; i < COLI_TQ_GROUP; i++)
        values[i] *= 0.08838834764831845f * coli_tq_s1[i];
}

static inline int coli_tq_nearest2(float value) {
    if (value < -0.086728f) return 0;
    if (value < 0.0f) return 1;
    if (value < 0.086728f) return 2;
    return 3;
}

static inline int coli_tq_nearest3(float value) {
    if (value < -0.154496f) return 0;
    if (value < -0.092804f) return 1;
    if (value < -0.044243f) return 2;
    if (value < 0.0f) return 3;
    if (value < 0.044243f) return 4;
    if (value < 0.092804f) return 5;
    if (value < 0.154496f) return 6;
    return 7;
}

static inline int coli_tq_nearest4(float value) {
    if (value < -0.212203f) return 0;
    if (value < -0.162947f) return 1;
    if (value < -0.127026f) return 2;
    if (value < -0.097164f) return 3;
    if (value < -0.070671f) return 4;
    if (value < -0.046174f) return 5;
    if (value < -0.022824f) return 6;
    if (value < 0.0f) return 7;
    if (value < 0.022824f) return 8;
    if (value < 0.046174f) return 9;
    if (value < 0.070671f) return 10;
    if (value < 0.097164f) return 11;
    if (value < 0.127026f) return 12;
    if (value < 0.162947f) return 13;
    if (value < 0.212203f) return 14;
    return 15;
}

static inline size_t coli_tq_block_bytes(int bits) {
    if (bits == 2) return sizeof(ColiTurbo2Block);
    if (bits == 3) return sizeof(ColiTurbo3Block);
    if (bits == 4) return sizeof(ColiTurbo4Block);
    return 0;
}

static inline int coli_tq_encode_group(void *destination, const float *source,
                                       int bits) {
    if (!destination || !source || !coli_tq_block_bytes(bits)) return -1;
    float values[COLI_TQ_GROUP];
    float norm_squared = 0.0f;
    for (int i = 0; i < COLI_TQ_GROUP; i++) {
        if (!isfinite(source[i])) return -1;
        values[i] = source[i];
        norm_squared += values[i] * values[i];
    }
    if (!isfinite(norm_squared)) return -1;
    float norm = sqrtf(norm_squared);
    float inverse_norm = norm > 0.0f ? 1.0f / norm : 0.0f;
    for (int i = 0; i < COLI_TQ_GROUP; i++) values[i] *= inverse_norm;
    coli_tq_fwht(values);

    float reconstruction_squared = 0.0f;
    ColiTurbo2Block block2 = {0};
    ColiTurbo3Block block3 = {0};
    ColiTurbo4Block block4 = {0};
    if (bits == 2) {
        for (int i = 0; i < COLI_TQ_GROUP; i++) {
            int index = coli_tq_nearest2(values[i]);
            block2.qs[i / 4] |= (uint8_t)(index << ((i % 4) * 2));
            reconstruction_squared += coli_tq_centroids2[index] *
                                      coli_tq_centroids2[index];
        }
    } else if (bits == 3) {
        for (int i = 0; i < COLI_TQ_GROUP; i++) {
            int index = coli_tq_nearest3(values[i]);
            block3.qs[i / 4] |= (uint8_t)((index & 3) << ((i % 4) * 2));
            if (index & 4) block3.signs[i / 8] |= (uint8_t)(1u << (i % 8));
            reconstruction_squared += coli_tq_centroids3[index] *
                                      coli_tq_centroids3[index];
        }
    } else {
        for (int i = 0; i < COLI_TQ_GROUP; i++) {
            int index = coli_tq_nearest4(values[i]);
            block4.qs[i / 2] |= (uint8_t)(index << ((i % 2) * 4));
            reconstruction_squared += coli_tq_centroids4[index] *
                                      coli_tq_centroids4[index];
        }
    }

    float reconstruction_norm = sqrtf(reconstruction_squared);
    float corrected = reconstruction_norm > 0.0f
        ? norm / reconstruction_norm : norm;
    if (!isfinite(corrected)) return -1;
    uint16_t encoded_norm = coli_tq_fp16_encode(corrected);
    if ((encoded_norm & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) return -1;
    if (bits == 2) {
        memcpy(&block2.norm, &encoded_norm, sizeof(encoded_norm));
        memcpy(destination, &block2, sizeof(block2));
    } else if (bits == 3) {
        memcpy(&block3.norm, &encoded_norm, sizeof(encoded_norm));
        memcpy(destination, &block3, sizeof(block3));
    } else {
        memcpy(&block4.norm, &encoded_norm, sizeof(encoded_norm));
        memcpy(destination, &block4, sizeof(block4));
    }
    return 0;
}

static inline int coli_tq_decode_group(float *destination, const void *source,
                                       int bits) {
    if (!destination || !source || !coli_tq_block_bytes(bits)) return -1;
    float norm;
    if (bits == 2) {
        ColiTurbo2Block stored;
        memcpy(&stored, source, sizeof(stored));
        const ColiTurbo2Block *block = &stored;
        norm = coli_tq_fp16_decode(block->norm);
        for (int i = 0; i < COLI_TQ_GROUP; i++) {
            int index = (block->qs[i / 4] >> ((i % 4) * 2)) & 3;
            destination[i] = coli_tq_centroids2[index] * norm;
        }
    } else if (bits == 3) {
        ColiTurbo3Block stored;
        memcpy(&stored, source, sizeof(stored));
        const ColiTurbo3Block *block = &stored;
        norm = coli_tq_fp16_decode(block->norm);
        for (int i = 0; i < COLI_TQ_GROUP; i++) {
            int low = (block->qs[i / 4] >> ((i % 4) * 2)) & 3;
            int high = (block->signs[i / 8] >> (i % 8)) & 1;
            destination[i] = coli_tq_centroids3[low | (high << 2)] * norm;
        }
    } else {
        ColiTurbo4Block stored;
        memcpy(&stored, source, sizeof(stored));
        const ColiTurbo4Block *block = &stored;
        norm = coli_tq_fp16_decode(block->norm);
        for (int i = 0; i < COLI_TQ_GROUP; i++) {
            int index = (block->qs[i / 2] >> ((i % 2) * 4)) & 15;
            destination[i] = coli_tq_centroids4[index] * norm;
        }
    }
    if (!isfinite(norm)) return -1;
    coli_tq_fwht_inverse(destination);
    return 0;
}

static inline int coli_tq_encode(void *destination, const float *source,
                                 int length, int bits) {
    size_t block_bytes = coli_tq_block_bytes(bits);
    if (!block_bytes || !destination || !source || length < 1 ||
        length % COLI_TQ_GROUP)
        return -1;
    unsigned char *output = (unsigned char *)destination;
    for (int base = 0; base < length; base += COLI_TQ_GROUP) {
        if (coli_tq_encode_group(output, source + base, bits)) return -1;
        output += block_bytes;
    }
    return 0;
}

static inline int coli_tq_decode(float *destination, const void *source,
                                 int length, int bits) {
    size_t block_bytes = coli_tq_block_bytes(bits);
    if (!block_bytes || !destination || !source || length < 1 ||
        length % COLI_TQ_GROUP)
        return -1;
    const unsigned char *input = (const unsigned char *)source;
    for (int base = 0; base < length; base += COLI_TQ_GROUP) {
        if (coli_tq_decode_group(destination + base, input, bits)) return -1;
        input += block_bytes;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif
