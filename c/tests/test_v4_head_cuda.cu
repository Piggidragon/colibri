#include "../backend_cuda_v4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t state = UINT32_C(0x6a09e667);

static float sample(void) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    return ((state >> 8) & 0xffff) / 32768.0f - 1.0f;
}

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (uint16_t)(bits >> 16);
}

static float decode(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int run_case(int vocab) {
    enum { DIMENSION = 4096 };
    size_t weights_bytes = (size_t)vocab * DIMENSION * sizeof(uint16_t);
    uint16_t *weights = (uint16_t *)malloc(weights_bytes);
    float *hidden = (float *)malloc((size_t)DIMENSION * sizeof(*hidden));
    float *cpu = (float *)malloc((size_t)vocab * sizeof(*cpu));
    float *gpu = (float *)malloc((size_t)vocab * sizeof(*gpu));
    if (!weights || !hidden || !cpu || !gpu) return -1;
    for (int column = 0; column < DIMENSION; column++) hidden[column] = sample();
    for (size_t item = 0; item < (size_t)vocab * DIMENSION; item++)
        weights[item] = bf16(sample());
    for (int row = 0; row < vocab; row++) {
        float sum = 0.0f;
        for (int column = 0; column < DIMENSION; column++)
            sum += decode(weights[(size_t)row * DIMENSION + column]) * hidden[column];
        cpu[row] = sum;
    }
    void *device = v4_cuda_kv_alloc(weights_bytes);
    int result = !device || v4_cuda_copy_to_device(device, 0, weights, weights_bytes)
        || v4_cuda_head_logits(gpu, device, hidden, 1, vocab, DIMENSION);
    float max_abs = 0.0f;
    double dot = 0.0, cpu2 = 0.0, gpu2 = 0.0;
    if (!result) for (int row = 0; row < vocab; row++) {
        float error = fabsf(cpu[row] - gpu[row]);
        if (error > max_abs) max_abs = error;
        dot += (double)cpu[row] * gpu[row];
        cpu2 += (double)cpu[row] * cpu[row];
        gpu2 += (double)gpu[row] * gpu[row];
    }
    float logit = 0.0f;
    int token = -1;
    if (!result) result = v4_cuda_head_argmax(&logit, &token, device, hidden,
                                               vocab, DIMENSION);
    int expected = 0;
    for (int row = 1; row < vocab; row++) if (cpu[row] > cpu[expected]) expected = row;
    double cosine = cpu2 && gpu2 ? dot / sqrt(cpu2 * gpu2) : 1.0;
    if (!result && (token != expected || max_abs > 2e-3f || cosine < 0.999999)) {
        fprintf(stderr, "head vocab=%d token=%d expected=%d max_abs=%g cosine=%.9f\n",
                vocab, token, expected, max_abs, cosine);
        result = -1;
    }
    v4_cuda_kv_free(device);
    free(gpu); free(cpu); free(hidden); free(weights);
    return result;
}

static int tie_breaks_to_lower_token(void) {
    enum { VOCAB = 128, DIMENSION = 4096 };
    size_t bytes = (size_t)VOCAB * DIMENSION * sizeof(uint16_t);
    uint16_t *weights = (uint16_t *)calloc(1, bytes);
    float *hidden = (float *)calloc(DIMENSION, sizeof(*hidden));
    if (!weights || !hidden) return -1;
    for (int column = 0; column < DIMENSION; column++) hidden[column] = 1.0f;
    for (int column = 0; column < DIMENSION; column++) {
        weights[(size_t)17 * DIMENSION + column] = bf16(1.0f);
        weights[(size_t)42 * DIMENSION + column] = bf16(1.0f);
    }
    void *device = v4_cuda_kv_alloc(bytes);
    float logit = 0.0f;
    int token = -1;
    int result = !device || v4_cuda_copy_to_device(device, 0, weights, bytes) ||
        v4_cuda_head_argmax(&logit, &token, device, hidden, VOCAB, DIMENSION) ||
        token != 17;
    v4_cuda_kv_free(device);
    free(hidden); free(weights);
    return result ? -1 : 0;
}

int main(void) {
    if (v4_cuda_init(0)) {
        puts("test_v4_head_cuda: skipped (no CUDA device)");
        return 77;
    }
    int result = run_case(128) || run_case(129280) || tie_breaks_to_lower_token();
    v4_cuda_shutdown();
    if (result) return 1;
    puts("test_v4_head_cuda: ok");
    return 0;
}
