#include "../deepseek_v4_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void set_chunk(const char *value) {
#ifdef _WIN32
    _putenv_s("V4_PREFILL_CHUNK", value ? value : "");
#else
    if (value) setenv("V4_PREFILL_CHUNK", value, 1);
    else unsetenv("V4_PREFILL_CHUNK");
#endif
}

static int expect_chunk(const char *label, const char *value, int capacity,
                        int expected) {
    set_chunk(value);
    int actual = coli_v4_prefill_chunk_tokens(capacity);
    if (actual == expected) return 0;
    fprintf(stderr, "%s: expected %d tokens, got %d\n", label, expected,
            actual);
    return 1;
}

int main(void) {
    int failed = 0;
    failed |= expect_chunk("default", NULL, 262144, 0);
    failed |= expect_chunk("off", "0", 262144, 0);
    failed |= expect_chunk("minimum clamp", "1", 262144, 64);
    failed |= expect_chunk("round down", "32799", 262144, 32768);
    failed |= expect_chunk("maximum clamp", "999999", 262144, 65536);
    failed |= expect_chunk("prompt cap", "32768", 4096, 4096);
    failed |= expect_chunk("garbage", "many", 262144, 0);
    failed |= expect_chunk("negative", "-1", 262144, 0);
    failed |= expect_chunk("overflow", "999999999999999999999", 262144, 0);

    set_chunk("32768");
    int slots = coli_v4_prefill_chunk_tokens(262144);
    uint64_t bytes = 0;
    if (coli_v4_session_state_bytes(slots, 4, 4096, &bytes) ||
        bytes != 4 * UINT64_C(1073741824)) {
        fprintf(stderr, "chunked state: expected 4 GiB, got %llu bytes\n",
                (unsigned long long)bytes);
        failed = 1;
    }
    set_chunk(NULL);
    if (!failed) puts("test_v4_prefill_chunk: ok");
    return failed ? 1 : 0;
}
