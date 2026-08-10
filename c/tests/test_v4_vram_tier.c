/* plans/08-vram-planner.md Commit 1: coli_v4_vram_tier_plan decides which of
 * the four VRAM-eligible tiers (KV, dense, head, DSpark) fit the card, in
 * fixed priority order, with the head<->DSpark coupling invariant from
 * plan 07 and an all-RAM answer (never an error) when there is no VRAM at
 * all. */
#include "../deepseek_v4_internal.h"

#include <stdint.h>
#include <stdio.h>

#define GIB UINT64_C(1073741824)

static int failures;

static void check(int condition, const char *what) {
    if (condition) {
        printf("vram tier: %s: ok\n", what);
        return;
    }
    fprintf(stderr, "vram tier: %s: FAILED\n", what);
    failures++;
}

static int plan_for(ColiV4VramTierPlan *plan, uint64_t vram_available,
                    uint64_t vram_reserve, uint64_t kv, uint64_t dense,
                    uint64_t head, uint64_t dspark, int dspark_wanted) {
    ColiV4VramTierInputs inputs = {
        vram_available, vram_reserve, kv, dense, head, dspark, dspark_wanted,
    };
    char error[256] = {0};
    if (coli_v4_vram_tier_plan(plan, &inputs, error, sizeof(error))) {
        fprintf(stderr, "vram tier plan failed: %s\n", error);
        return -1;
    }
    return 0;
}

int main(void) {
    ColiV4VramTierPlan plan;

    /* Alles passt. */
    if (plan_for(&plan, 12 * GIB, 1 * GIB, 1 * GIB, 5 * GIB, 1 * GIB,
                1 * GIB, 1))
        return 1;
    check(plan.kv == COLI_V4_TIER_VRAM && plan.dense == COLI_V4_TIER_VRAM &&
          plan.head == COLI_V4_TIER_VRAM && plan.dspark == COLI_V4_TIER_VRAM &&
          plan.vram_used_bytes == 8 * GIB,
          "everything fits lands on the card");

    /* Knapp: KV + Dense passen, Head nicht -> head UND dspark auf RAM
     * (Kopplungsinvariante), obwohl dspark allein locker passen würde. */
    if (plan_for(&plan, 8 * GIB, 1 * GIB, 1 * GIB, 5 * GIB, 5 * GIB,
                (uint64_t)(0.1 * GIB), 1))
        return 1;
    check(plan.kv == COLI_V4_TIER_VRAM && plan.dense == COLI_V4_TIER_VRAM,
          "kv and dense still admitted when head does not fit");
    check(plan.head == COLI_V4_TIER_RAM, "head that does not fit falls back to RAM");
    check(plan.dspark == COLI_V4_TIER_RAM,
          "dspark follows head to RAM even though its own bytes would fit");

    /* Sehr knapp: nur KV passt. */
    if (plan_for(&plan, 2 * GIB, 1 * GIB, (uint64_t)(0.5 * GIB), 5 * GIB,
                1 * GIB, (uint64_t)(0.1 * GIB), 1))
        return 1;
    check(plan.kv == COLI_V4_TIER_VRAM, "kv alone still fits a very tight card");
    check(plan.dense == COLI_V4_TIER_RAM && plan.head == COLI_V4_TIER_RAM &&
          plan.dspark == COLI_V4_TIER_RAM,
          "everything below kv falls back when only kv fits");

    /* Kein GPU im System: kein Fehler, alles RAM. */
    if (plan_for(&plan, 0, 0, (uint64_t)(0.5 * GIB), 5 * GIB, 1 * GIB,
                (uint64_t)(0.1 * GIB), 1))
        return 1;
    check(plan.kv == COLI_V4_TIER_RAM && plan.dense == COLI_V4_TIER_RAM &&
          plan.head == COLI_V4_TIER_RAM && plan.dspark == COLI_V4_TIER_RAM &&
          plan.vram_used_bytes == 0,
          "no VRAM in the system degrades to an all-RAM plan without erroring");

    /* dspark_wanted=0: DSpark never claims VRAM even though it would fit and
     * the head is resident -- matches the "off" report state, not a tier
     * that happened to lose a fight for budget. */
    if (plan_for(&plan, 12 * GIB, 1 * GIB, 1 * GIB, 5 * GIB, 1 * GIB,
                1 * GIB, 0))
        return 1;
    check(plan.head == COLI_V4_TIER_VRAM && plan.dspark == COLI_V4_TIER_RAM,
          "dspark_wanted=0 keeps dspark off the card regardless of budget");

    /* Reserve clamp: a reserve at or above the free bytes leaves a zero
     * budget, not an underflowed one. */
    if (plan_for(&plan, 1 * GIB, 1 * GIB, 1, 0, 0, 0, 0))
        return 1;
    check(plan.kv == COLI_V4_TIER_RAM, "reserve equal to free bytes leaves no budget");
    if (plan_for(&plan, 1 * GIB, 2 * GIB, 1, 0, 0, 0, 0))
        return 1;
    check(plan.kv == COLI_V4_TIER_RAM,
          "reserve above free bytes clamps to zero budget instead of underflowing");

    /* Overflow guard: UINT64_MAX-sized asks never wrap into a false fit. */
    if (plan_for(&plan, 12 * GIB, 1 * GIB, UINT64_MAX, UINT64_MAX,
                UINT64_MAX, UINT64_MAX, 1))
        return 1;
    check(plan.kv == COLI_V4_TIER_RAM && plan.vram_used_bytes == 0,
          "UINT64_MAX-sized tiers never wrap around into a spurious fit");

    if (failures) {
        fprintf(stderr, "V4 VRAM tier tests: %d failed\n", failures);
        return 1;
    }
    puts("V4 VRAM tier tests: ok");
    return 0;
}
