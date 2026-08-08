#include "../v4_kv_codec.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    int ratios[43] = {0};
    for (int layer = 2; layer < 43; layer++)
        ratios[layer] = layer % 2 ? 128 : 4;
    uint64_t f32 = coli_v4_kv_context_bytes(
        43, 128, 512, 64, 128, ratios, 131072,
        COLI_V4_KV_F32, COLI_V4_KV_F32);
    uint64_t native = coli_v4_kv_context_bytes(
        43, 128, 512, 64, 128, ratios, 131072,
        COLI_V4_KV_NATIVE, COLI_V4_KV_NATIVE);
    uint64_t expected_f32 = UINT64_C(43) * 128 * 2048;
    uint64_t expected_native = UINT64_C(43) * 128 * 583;
    for (int layer = 2; layer < 43; layer++) {
        uint64_t rows = (UINT64_C(131072) + ratios[layer] - 1) / ratios[layer];
        expected_f32 += rows * 2048;
        expected_native += rows * 583;
        if (ratios[layer] == 4) {
            expected_f32 += rows * 512;
            expected_native += rows * 68;
        }
    }
    if (f32 != expected_f32 || native != expected_native || native >= f32) {
        fprintf(stderr, "V4 context byte accounting mismatch\n");
        return 1;
    }
    int no_compression = 0;
    if (coli_v4_kv_context_bytes(
            INT_MAX, INT_MAX, INT_MAX, 0, 1, &no_compression, 1,
            COLI_V4_KV_F32, COLI_V4_KV_F32) != UINT64_MAX) {
        fprintf(stderr, "V4 sliding-window byte overflow was not rejected\n");
        return 1;
    }
    printf("V4 KV context 128k: f32=%.3f GiB native=%.3f GiB saved=%.3f GiB\n",
           f32 / 1073741824.0, native / 1073741824.0,
           (f32 - native) / 1073741824.0);
    return 0;
}
