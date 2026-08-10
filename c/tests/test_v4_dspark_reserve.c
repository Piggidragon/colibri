/* Phase 07: the DSpark host reserve must be booked before the expert store is
 * sized, and it must not shrink when the backbone is headed for VRAM.  The
 * upload is transactional -- the host holds every FP8 tensor until the last
 * V4_DS_UPLOAD succeeds, and the Markov tensors stay resident either way -- so
 * a smaller VRAM-mode reserve would let the first lazy draft borrow its upload
 * peak from the target cache.  Built with -DCOLI_V4_TEST_HOOKS against the
 * objects under build/ownership/. */
#include "../deepseek_v4_internal.h"
#include "v4_engine_fixture.h"

#include <stdio.h>
#include <stdlib.h>

#define MIB UINT64_C(1048576)

/* Stubs for symbols referenced by RUNTIME but unused here. */
int coli_v4_expert_store_open_planned(
    ColiV4Engine *engine,
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size) {
    (void)engine;
    (void)options;
    (void)store;
    if (error && error_size)
        snprintf(error, error_size, "expert store stub");
    return -1;
}

void coli_v4_layer_resident_reference_free(
    ColiV4Engine *engine, ColiDeepSeekV4LayerWeights *weights) {
    (void)engine;
    (void)weights;
}

/* Opens an engine with the given knobs and reports its booked reserve. */
static int reserve_for(const char *mtp, const char *draft, const char *cache_gb,
                       const char *vram, uint64_t *reserve) {
    char directory[128], error[256];
    if (v4_fixture_make(directory, sizeof(directory), "colibri-v4-dsres-XXXXXX"))
        return -1;
    if (mtp) setenv("V4_MTP", mtp, 1); else unsetenv("V4_MTP");
    if (draft) setenv("V4_DRAFT", draft, 1); else unsetenv("V4_DRAFT");
    if (cache_gb) setenv("V4_MTP_GB", cache_gb, 1); else unsetenv("V4_MTP_GB");
    if (vram) setenv("V4_VRAM", vram, 1); else unsetenv("V4_VRAM");

    coli_v4_test_fail_expert_store_open = 0;
    coli_v4_test_skip_expert_store_open = 1;
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions options = {.target_model_dir = directory};
    int failed = coli_v4_engine_open(&engine, &options, error, sizeof(error));
    if (!failed) {
        *reserve = engine->runtime.dspark_reserve_bytes;
        coli_v4_engine_destroy(engine);
    } else {
        fprintf(stderr, "engine open failed: %s\n", error);
    }
    coli_v4_test_skip_expert_store_open = 0;
    unsetenv("V4_MTP");
    unsetenv("V4_DRAFT");
    unsetenv("V4_MTP_GB");
    unsetenv("V4_VRAM");
    v4_fixture_cleanup(directory);
    return failed ? -1 : 0;
}

static int expect(const char *label, uint64_t actual, uint64_t wanted) {
    if (actual == wanted) {
        printf("dspark reserve: %-34s %.3f GiB: ok\n", label,
               actual / 1073741824.0);
        return 0;
    }
    fprintf(stderr, "dspark reserve: %s expected %llu but got %llu\n", label,
            (unsigned long long)wanted, (unsigned long long)actual);
    return 1;
}

int main(void) {
    uint64_t host = 0, device = 0, wide = 0, off = 0, clamped = 0;
    if (reserve_for("1", "3", NULL, "0", &host) ||
        reserve_for("1", "3", NULL, "1", &device) ||
        reserve_for("1", "3", "1.0", "0", &wide) ||
        reserve_for("1", "3", "9999", "0", &clamped) ||
        reserve_for("0", "0", NULL, "0", &off))
        return 1;

    /* The documented default: 0.45 GB cache plus the 768 MiB upload peak. */
    int result = expect("default V4_MTP_GB=0.45",
                        host, (uint64_t)(0.45 * 1e9) + 768 * MIB);
    /* In a non-CUDA build V4_VRAM=1 degrades to vram_enabled=0, so this pins
     * the request path only.  That the **resident*** backbone books the same
     * reserve is a property of v4_dspark_full_reserve_bytes ignoring its
     * argument, which test_deepseek_v4_dspark_source.py guards at the source. */
    result |= expect("V4_VRAM=1 requested books the same", device, host);
    result |= expect("V4_MTP_GB=1.0", wide, UINT64_C(1000000000) + 768 * MIB);
    result |= expect("V4_MTP_GB clamped to 4.0",
                     clamped, UINT64_C(4000000000) + 768 * MIB);
    result |= expect("drafting off books nothing", off, 0);
    if (result) return 1;
    puts("DeepSeek-V4 DSpark reserve tests: ok");
    return 0;
}
