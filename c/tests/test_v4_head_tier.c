/* Phase 07: the BF16 head lives in exactly one place.  Once it is uploaded the
 * host buffer is released, so every host-side reader must see NULL rather than
 * a stale pointer -- the plan calls this the branch's subtlest coupling, since
 * a head that silently keeps a host copy costs the RAM the phase set out to
 * free, and one that loses both leaves readers dereferencing NULL. */
#include "../deepseek_v4_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEAD_OFFSET UINT64_C(4096)
#define HEAD_BYTES  UINT64_C(2048)
#define HEAD_SHARD  3

static int failures;

static void check(int condition, const char *what) {
    if (condition) {
        printf("head tier: %s: ok\n", what);
        return;
    }
    fprintf(stderr, "head tier: %s: FAILED\n", what);
    failures++;
}

/* Both accessors apply the same window; only the tier they answer for differs. */
static void check_window(const ColiV4Engine *engine, const char *tier,
                         const void *(*get)(const ColiV4Engine *, int,
                                            uint64_t, size_t),
                         int resident) {
    char label[128];
    snprintf(label, sizeof(label), "%s %s its window", tier,
             resident ? "serves" : "refuses");
    check((get(engine, HEAD_SHARD, HEAD_OFFSET, HEAD_BYTES) != NULL) == resident,
          label);
    snprintf(label, sizeof(label), "%s rejects a foreign shard", tier);
    check(get(engine, HEAD_SHARD + 1, HEAD_OFFSET, HEAD_BYTES) == NULL, label);
    snprintf(label, sizeof(label), "%s rejects an offset below the window", tier);
    check(get(engine, HEAD_SHARD, HEAD_OFFSET - 1, 1) == NULL, label);
    snprintf(label, sizeof(label), "%s rejects a length past the window", tier);
    check(get(engine, HEAD_SHARD, HEAD_OFFSET, HEAD_BYTES + 1) == NULL, label);
}

int main(void) {
    ColiV4Engine *engine = calloc(1, sizeof(*engine));
    unsigned char *host = calloc(1, HEAD_BYTES);
    if (!engine || !host) {
        fprintf(stderr, "head tier: allocation failed\n");
        free(host);
        free(engine);
        return 1;
    }
    /* A device pointer is never dereferenced here, only compared and offset. */
    void *device = (void *)(uintptr_t)0x1000;

    engine->head_cache.bytes = HEAD_BYTES;
    engine->head_cache.offset = HEAD_OFFSET;
    engine->head_cache.shard = HEAD_SHARD;

    engine->head_cache.data = host;
    engine->head_cache.device = NULL;
    check_window(engine, "resident head", coli_v4_head_cache_data, 1);
    check_window(engine, "resident head device view",
                 coli_v4_head_cache_device, 0);
    check(coli_v4_head_cache_bytes(engine) == HEAD_BYTES,
          "resident head reports its size");

    engine->head_cache.data = NULL;
    engine->head_cache.device = device;
    check_window(engine, "device head", coli_v4_head_cache_device, 1);
    check_window(engine, "device head host view", coli_v4_head_cache_data, 0);
    check(coli_v4_head_cache_bytes(engine) == HEAD_BYTES,
          "device head reports its size");

    /* The tier that was never populated must answer NULL for both. */
    engine->head_cache.data = NULL;
    engine->head_cache.device = NULL;
    check(coli_v4_head_cache_data(engine, HEAD_SHARD, HEAD_OFFSET,
                                  HEAD_BYTES) == NULL &&
          coli_v4_head_cache_device(engine, HEAD_SHARD, HEAD_OFFSET,
                                    HEAD_BYTES) == NULL,
          "streamed head serves neither tier");
    check(coli_v4_head_cache_data(NULL, HEAD_SHARD, HEAD_OFFSET, 1) == NULL &&
          coli_v4_head_cache_device(NULL, HEAD_SHARD, HEAD_OFFSET, 1) == NULL,
          "a NULL engine is not a cache hit");

    free(host);
    free(engine);
    if (failures) {
        fprintf(stderr, "DeepSeek-V4 head tier tests: %d failed\n", failures);
        return 1;
    }
    puts("DeepSeek-V4 head tier tests: ok");
    return 0;
}
