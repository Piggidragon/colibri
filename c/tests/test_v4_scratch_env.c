#include "../deepseek_v4_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MIB UINT64_C(1048576)

static void set_scratch(const char *value) {
#ifdef _WIN32
    _putenv_s("V4_SCRATCH_MB", value);
#else
    setenv("V4_SCRATCH_MB", value, 1);
#endif
}

static void clear_scratch(void) {
#ifdef _WIN32
    _putenv_s("V4_SCRATCH_MB", "");
#else
    unsetenv("V4_SCRATCH_MB");
#endif
}

static void set_context(const char *value) {
#ifdef _WIN32
    _putenv_s("CTX", value);
#else
    setenv("CTX", value, 1);
#endif
}

static void clear_context(void) {
#ifdef _WIN32
    _putenv_s("CTX", "");
#else
    unsetenv("CTX");
#endif
}

static int expect_mb(const char *label, const char *value, uint64_t expected) {
    if (value) set_scratch(value); else clear_scratch();
    uint64_t actual = coli_v4_scratch_bytes() / MIB;
    if (actual == expected) return 0;
    fprintf(stderr, "%s: expected %llu MiB, got %llu MiB\n", label,
            (unsigned long long)expected, (unsigned long long)actual);
    return 1;
}

static int expect_context(const char *label, const char *value, int expected) {
    if (value) set_context(value); else clear_context();
    int actual = coli_v4_context_tokens();
    if (actual == expected) return 0;
    fprintf(stderr, "%s: expected %d tokens, got %d\n", label,
            expected, actual);
    return 1;
}

int main(void) {
    int failed = 0;
    failed |= expect_mb("default", NULL, 512);
    failed |= expect_mb("explicit", "128", 128);
    failed |= expect_mb("lower clamp", "1", 64);
    failed |= expect_mb("upper clamp", "999999", 4096);
    failed |= expect_mb("garbage", "abc", 512);
    failed |= expect_mb("trailing garbage", "128MB", 512);
    failed |= expect_mb("overflow", "999999999999999999999999", 512);
    failed |= expect_mb("empty", "", 512);
    failed |= expect_context("context default", NULL, 4096);
    failed |= expect_context("context explicit", "131072", 131072);
    failed |= expect_context("context too small", "1", 4096);
    failed |= expect_context("context garbage", "128k", 4096);
    failed |= expect_context("context overflow",
                             "999999999999999999999999", 4096);
    failed |= expect_context("context above int", "2147483648", 4096);
    uint64_t session_bytes = 0;
    if (coli_v4_session_state_bytes(131072, 2, 2048, &session_bytes) ||
        session_bytes != 4 * UINT64_C(1073741824)) {
        fprintf(stderr, "session state: expected 4 GiB, got %llu bytes\n",
                (unsigned long long)session_bytes);
        failed = 1;
    }
    clear_scratch();
    clear_context();
    if (!failed) puts("test_v4_scratch_env: ok");
    return failed ? 1 : 0;
}
