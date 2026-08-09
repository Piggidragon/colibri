#include "../deepseek_v4_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { ROWS = 8, COLUMNS = 128, BYTES = ROWS * COLUMNS };

static void fill(unsigned char *data) {
    for (int row = 0; row < ROWS; row++)
        for (int column = 0; column < COLUMNS; column++)
            data[(size_t)row * COLUMNS + column] =
                (unsigned char)(row * 29 + column);
}

int main(void) {
    unsigned char original[BYTES], gpu[BYTES], cpu[BYTES];
    fill(original);
    memcpy(gpu, original, sizeof(gpu));
    memcpy(cpu, original, sizeof(cpu));

    int gpu_packed = coli_v4_test_fp8_maybe_pack_rows8(
        gpu, ROWS, COLUMNS, 1);
    if (gpu_packed != 0 || memcmp(gpu, original, sizeof(gpu)) != 0) {
        fprintf(stderr, "GPU-resident FP8 was rows8-packed\n");
        return 1;
    }

    int cpu_packed = coli_v4_test_fp8_maybe_pack_rows8(
        cpu, ROWS, COLUMNS, 0);
#ifdef __AVX2__
    if (cpu_packed != 1) {
        fprintf(stderr, "AVX2 FP8 did not report rows8 packing\n");
        return 1;
    }
    for (int column = 0; column < COLUMNS; column++)
        for (int row = 0; row < ROWS; row++)
            if (cpu[(size_t)column * ROWS + row] !=
                original[(size_t)row * COLUMNS + column]) {
                fprintf(stderr, "AVX2 rows8 layout mismatch\n");
                return 1;
            }
#else
    if (cpu_packed != 0 || memcmp(cpu, original, sizeof(cpu)) != 0) {
        fprintf(stderr, "scalar FP8 path unexpectedly changed bytes\n");
        return 1;
    }
#endif
    puts("V4 rows8 GPU skip tests: ok");
    return 0;
}
