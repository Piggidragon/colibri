#include "../deepseek_v4_internal.h"

#include <stdio.h>
#include <stdlib.h>

static void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

/* plans/12-expert-cache-policy.md Commit 1: V4_PIN_SLOTS and V4_PIN_FRACTION
 * replace the compile-time COLI_V4_MAX_PIN_SLOTS_PER_LAYER ceiling at
 * runtime.  V4_PIN_RAMP_REQUESTS likewise replaces COLI_V4_PIN_RAMP_REQUESTS.
 * The Makefile rule for this test pins both compile-time defines to the
 * values Makefile.deepseek-v4 actually ships (16, 24), so this exercises the
 * real production default, not the in-source #ifndef fallback (4, 0).
 * V4_PIN_SLOTS wins if both env knobs are set; either falls back to the
 * compile-time default on garbage input, and the result is always clamped
 * into [0, available_pins]. */
static int expect_ceiling(const char *label, const char *slots,
                          const char *fraction, int available_pins,
                          int expected) {
    set_env("V4_PIN_SLOTS", slots);
    set_env("V4_PIN_FRACTION", fraction);
    int actual = coli_v4_pin_slots_ceiling(available_pins);
    if (actual == expected) return 0;
    fprintf(stderr, "%s: expected %d, got %d\n", label, expected, actual);
    return 1;
}

static int expect_ramp(const char *label, const char *value, int expected) {
    set_env("V4_PIN_RAMP_REQUESTS", value);
    int actual = coli_v4_pin_ramp_requests();
    if (actual == expected) return 0;
    fprintf(stderr, "%s: expected %d, got %d\n", label, expected, actual);
    return 1;
}

int main(void) {
    int failed = 0;

    /* Default: neither knob set -> compile-time ceiling (16), clamped to
     * whatever is actually available. */
    failed |= expect_ceiling("default under ceiling", NULL, NULL, 10, 10);
    failed |= expect_ceiling("default at ceiling", NULL, NULL, 16, 16);
    failed |= expect_ceiling("default above ceiling", NULL, NULL, 44, 16);

    /* V4_PIN_SLOTS alone: absolute, still clamped to available_pins. */
    failed |= expect_ceiling("slots explicit", "32", NULL, 44, 32);
    failed |= expect_ceiling("slots above available", "32", NULL, 22, 22);
    failed |= expect_ceiling("slots zero", "0", NULL, 44, 0);
    failed |= expect_ceiling("slots garbage", "abc", NULL, 44, 16);
    failed |= expect_ceiling("slots negative", "-1", NULL, 44, 16);
    failed |= expect_ceiling("slots trailing garbage", "8x", NULL, 44, 16);

    /* V4_PIN_FRACTION alone: share of available_pins, rounded to nearest. */
    failed |= expect_ceiling("fraction half", NULL, "0.5", 44, 22);
    failed |= expect_ceiling("fraction zero", NULL, "0", 44, 0);
    failed |= expect_ceiling("fraction one", NULL, "1", 44, 44);
    failed |= expect_ceiling("fraction above one", NULL, "1.5", 44, 16);
    failed |= expect_ceiling("fraction negative", NULL, "-0.1", 44, 16);
    failed |= expect_ceiling("fraction garbage", NULL, "half", 44, 16);

    /* Both set: V4_PIN_SLOTS is applied last and wins. */
    failed |= expect_ceiling("both set, slots wins", "8", "1.0", 44, 8);
    failed |= expect_ceiling("both set, slots garbage falls to fraction",
                             "abc", "0.5", 44, 22);

    /* available_pins itself can be zero (tiny expert count) or negative
     * (shouldn't happen, but a bad caller must not underflow the result). */
    failed |= expect_ceiling("no available pins", "16", NULL, 0, 0);
    failed |= expect_ceiling("negative available pins", "16", NULL, -1, 0);

    /* Ramp requests. */
    failed |= expect_ramp("ramp default", NULL, 24);
    failed |= expect_ramp("ramp explicit", "100", 100);
    failed |= expect_ramp("ramp zero disables", "0", 0);
    failed |= expect_ramp("ramp garbage", "abc", 24);
    failed |= expect_ramp("ramp negative", "-1", 24);
    failed |= expect_ramp("ramp trailing garbage", "100x", 24);

    set_env("V4_PIN_SLOTS", NULL);
    set_env("V4_PIN_FRACTION", NULL);
    set_env("V4_PIN_RAMP_REQUESTS", NULL);
    if (!failed) puts("test_v4_pin_env: ok");
    return failed ? 1 : 0;
}
