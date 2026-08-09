#include "../deepseek_v4_internal.h"

#include <stdint.h>
#include <stdio.h>

#define GIB UINT64_C(1073741824)

static int expect_reserved(ColiDeepSeekV4DenseLocation location,
                           uint64_t available, uint64_t vram_available,
                           uint64_t vram_reserve,
                           uint64_t expected_host, uint64_t expected_device) {
    ColiDeepSeekV4ResidentTierInputs inputs = {
        .available_bytes = available,
        .vram_available_bytes = vram_available,
        .vram_reserve_bytes = vram_reserve,
        .fixed_bytes = 4 * GIB,
        .dense_bytes = 24 * GIB,
        .dense_device_bytes = 6 * GIB,
        .minimum_expert_bytes = 12 * GIB,
        .vram_enabled = 1,
    };
    ColiDeepSeekV4ResidentTierPlan plan;
    char error[256] = {0};
    if (coli_v4_resident_tier_plan(&plan, &inputs, error, sizeof(error))) {
        fprintf(stderr, "tier plan failed: %s\n", error);
        return 1;
    }
    if (plan.dense_location != location ||
        plan.dense_bytes != expected_host ||
        plan.dense_device_bytes != expected_device ||
        plan.dense_resident != (location != COLI_V4_DENSE_STREAMED)) {
        fprintf(stderr,
                "tier mismatch: location=%d host=%llu device=%llu resident=%d\n",
                (int)plan.dense_location,
                (unsigned long long)plan.dense_bytes,
                (unsigned long long)plan.dense_device_bytes,
                plan.dense_resident);
        return 1;
    }
    return 0;
}

static int expect(ColiDeepSeekV4DenseLocation location,
                  uint64_t available, uint64_t vram_available,
                  uint64_t expected_host, uint64_t expected_device) {
    return expect_reserved(location, available, vram_available, 0,
                           expected_host, expected_device);
}

int main(void) {
    if (expect(COLI_V4_DENSE_VRAM, 40 * GIB, 6 * GIB, 18 * GIB, 6 * GIB) ||
        expect(COLI_V4_DENSE_RAM, 40 * GIB, 5 * GIB, 24 * GIB, 0) ||
        expect(COLI_V4_DENSE_STREAMED, 39 * GIB, 5 * GIB, 0, 0) ||
        /* The reserve has to come off the top: 6 GiB dense no longer fits in
         * 6 GiB of free VRAM once the KV mirror is accounted for. */
        expect_reserved(COLI_V4_DENSE_RAM, 40 * GIB, 6 * GIB, 1 * GIB,
                        24 * GIB, 0) ||
        expect_reserved(COLI_V4_DENSE_VRAM, 40 * GIB, 7 * GIB, 1 * GIB,
                        18 * GIB, 6 * GIB))
        return 1;
    puts("V4 dense tier tests: ok");
    return 0;
}
