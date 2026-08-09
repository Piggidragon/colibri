#include "../backend_cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static float e4m3(uint8_t byte) {
    int sign = byte >> 7;
    int exponent = (byte >> 3) & 15;
    int mantissa = byte & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float value = exponent
        ? ldexpf(1.0f + mantissa / 8.0f, exponent - 7)
        : ldexpf(mantissa / 8.0f, -6);
    return sign ? -value : value;
}

static uint8_t weight_byte(size_t index) {
    uint8_t byte = (uint8_t)((index * 29 + 17) & 255);
    if ((byte & 0x7f) == 0x7f) byte ^= 1;
    return byte;
}

typedef struct {
    const char *name;
    int rows;
    int columns;
    int row_start;
    int output_rows;
} Shape;

static int check_shape(const Shape *shape) {
    size_t weight_count = (size_t)shape->rows * shape->columns;
    int scale_columns = (shape->columns + 127) / 128;
    size_t scale_count = (size_t)((shape->rows + 127) / 128) * scale_columns;
    uint8_t *weights = (uint8_t *)malloc(weight_count);
    float *scales = (float *)malloc(scale_count * sizeof(*scales));
    float *input = (float *)calloc((size_t)shape->columns, sizeof(*input));
    float *output = (float *)malloc((size_t)shape->output_rows * sizeof(*output));
    if (!weights || !scales || !input || !output) {
        free(output); free(input); free(scales); free(weights);
        fprintf(stderr, "%s: host allocation failed\n", shape->name);
        return 1;
    }
    for (size_t i = 0; i < weight_count; i++) weights[i] = weight_byte(i);
    for (size_t i = 0; i < scale_count; i++)
        scales[i] = ldexpf(1.0f + (float)(i % 7) / 16.0f, -8);

    int positions[4] = {0, shape->columns > 127 ? 127 : shape->columns - 1,
                        shape->columns > 128 ? 128 : shape->columns - 1,
                        shape->columns - 1};
    float values[4] = {0.25f, -0.5f, 0.75f, -0.125f};
    int unique = 0;
    for (int i = 0; i < 4; i++) {
        int seen = 0;
        for (int j = 0; j < unique; j++) seen |= positions[j] == positions[i];
        if (seen) continue;
        positions[unique] = positions[i];
        values[unique] = values[i];
        input[positions[unique]] = values[unique];
        unique++;
    }

    ColiCudaTensor *tensor = NULL;
    int failed = !coli_cuda_tensor_upload(
        &tensor, weights, scales, 8, shape->columns, shape->rows, 0) ||
        !coli_cuda_fp8_matmul_rows(tensor, output, input, 1,
                                   shape->row_start, shape->output_rows);
    int mismatches = 0;
    if (!failed) {
        for (int local_row = 0; local_row < shape->output_rows; local_row++) {
            int row = shape->row_start + local_row;
            double expected = 0.0;
            for (int i = 0; i < unique; i++) {
                int column = positions[i];
                size_t wi = (size_t)row * shape->columns + column;
                size_t si = (size_t)(row / 128) * scale_columns + column / 128;
                expected += (double)e4m3(weights[wi]) * input[column] * scales[si];
            }
            float tolerance = 1e-3f * (fabsf((float)expected) + 1e-3f);
            if (fabsf(output[local_row] - (float)expected) > tolerance)
                mismatches++;
        }
    }
    printf("dense-fp8 %-22s [%d,%d] rows=%d..%d: %s (%d mismatches)\n",
           shape->name, shape->rows, shape->columns, shape->row_start,
           shape->row_start + shape->output_rows - 1,
           failed ? "dispatch failed" : mismatches ? "FAIL" : "ok",
           mismatches);
    coli_cuda_tensor_free(tensor);
    free(output); free(input); free(scales); free(weights);
    return failed || mismatches;
}

int main(void) {
    int device = 0;
    if (!coli_cuda_init(&device, 1)) {
        fprintf(stderr, "V4 dense CUDA test skipped: CUDA unavailable\n");
        return 77;
    }
    float lut[256];
    for (int i = 0; i < 256; i++) lut[i] = e4m3((uint8_t)i);
    if (!coli_cuda_fp8_set_lut(lut)) return 1;

    const Shape shapes[] = {
        {"attn.wkv", 512, 4096, 0, 512},
        {"attn.wq_a", 1024, 4096, 0, 1024},
        {"attn.wq_b", 32768, 1024, 0, 32768},
        {"attn.wo_a/slice", 8192, 4096, 1024, 1024},
        {"attn.wo_b", 4096, 8192, 0, 4096},
        {"ffn.shared.w1/w3", 2048, 4096, 0, 2048},
        {"ffn.shared.w2", 4096, 2048, 0, 4096},
        {"attn.indexer.wq_b", 8192, 1024, 0, 8192},
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        failed |= check_shape(&shapes[i]);
    coli_cuda_shutdown();
    return failed ? 1 : 0;
}
