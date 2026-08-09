#include "../deepseek_v4_internal.h"

#include <stdio.h>

int coli_v4_compressor_snapshot_create(
    const ColiDeepSeekV4CompressorState *state,
    ColiV4CompressorSnapshot **output) {
    (void)state; (void)output;
    return -1;
}

int coli_v4_compressor_snapshot_restore(
    ColiDeepSeekV4CompressorState *state,
    const ColiV4CompressorSnapshot *snapshot) {
    (void)state; (void)snapshot;
    return -1;
}

void coli_v4_compressor_snapshot_destroy(ColiV4CompressorSnapshot *snapshot) {
    (void)snapshot;
}

int main(void) {
    if (coli_v4_test_attention_encode_rejection_error() ||
        coli_v4_test_attention_snapshot_roundtrip(COLI_V4_KV_F32) ||
        coli_v4_test_attention_snapshot_roundtrip(COLI_V4_KV_NATIVE)) {
        fprintf(stderr, "V4 KV snapshot round-trip failed\n");
        return 1;
    }
    if (coli_v4_test_indexer_snapshot_rejections()) {
        fprintf(stderr, "V4 indexer snapshot rejection failed\n");
        return 1;
    }
    puts("V4 KV snapshot round-trips passed");
    return 0;
}
