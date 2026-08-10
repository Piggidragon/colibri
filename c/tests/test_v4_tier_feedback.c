/* plans/08-vram-planner.md Commit 1: the priority order (KV > dense > head >
 * DSpark) is only real if a higher-priority tier's claim actually reduces
 * what is left for the tiers below it -- the exact bug the pre-08 code had
 * for KV, which never budgeted anything at all and let dense/head/DSpark
 * claim the whole card first (plans/08-vram-planner.md, "Rückkopplung in den
 * RAM-Plan"). This file pins that feedback down as a monotonicity property,
 * rather than one fixed before/after pair. */
#include "../deepseek_v4_internal.h"

#include <stdint.h>
#include <stdio.h>

#define GIB UINT64_C(1073741824)

static int failures;

static void check(int condition, const char *what) {
    if (condition) {
        printf("tier feedback: %s: ok\n", what);
        return;
    }
    fprintf(stderr, "tier feedback: %s: FAILED\n", what);
    failures++;
}

int main(void) {
    /* A card that comfortably fits dense on its own (5 GiB dense against a
     * 10 GiB budget), but not once KV also wants a growing share.  Priority
     * says KV must never lose this fight, and dense must be the one to give
     * way -- exactly the "dense unconditionally wins" bug 08 fixes. */
    const uint64_t available = 10 * GIB, reserve = 1 * GIB, dense = 5 * GIB;
    uint64_t steps[] = {0, (uint64_t)(1.5 * GIB), 3 * GIB, 5 * GIB, 8 * GIB};
    int dense_was_vram = -1;
    for (size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); i++) {
        ColiV4VramTierInputs inputs = {
            available, reserve, steps[i], dense, 0, 0, 0,
        };
        ColiV4VramTierPlan plan;
        char error[256] = {0};
        if (coli_v4_vram_tier_plan(&plan, &inputs, error, sizeof(error))) {
            fprintf(stderr, "vram tier plan failed: %s\n", error);
            return 1;
        }
        check(plan.kv == COLI_V4_TIER_VRAM,
              "kv is never displaced by a lower-priority tier");
        /* Once dense has been pushed to RAM by a larger KV ask, a further
         * increase must never pull it back onto the card -- growth in a
         * higher-priority tier's claim only ever costs lower-priority tiers
         * budget, it never returns any. */
        if (dense_was_vram == 0)
            check(plan.dense == COLI_V4_TIER_RAM,
                  "dense does not un-lose a budget fight as kv grows further");
        dense_was_vram = plan.dense == COLI_V4_TIER_VRAM;
        check(plan.vram_used_bytes <= available - reserve,
              "used bytes never exceed the budget the reserve left behind");
        check(plan.vram_used_bytes >= steps[i] || plan.kv != COLI_V4_TIER_VRAM,
              "a resident kv tier is always counted in used bytes");
    }
    check(dense_was_vram == 0,
          "growing kv eventually pushes dense off the card entirely");

    /* The same budget with kv=0 must let dense through -- confirms the
     * displacement above is caused by kv's claim, not some other change. */
    {
        ColiV4VramTierInputs inputs = {available, reserve, 0, dense, 0, 0, 0};
        ColiV4VramTierPlan plan;
        char error[256] = {0};
        if (coli_v4_vram_tier_plan(&plan, &inputs, error, sizeof(error)))
            return 1;
        check(plan.dense == COLI_V4_TIER_VRAM,
              "dense fits fine once kv is not competing for the budget");
    }

    if (failures) {
        fprintf(stderr, "V4 tier feedback tests: %d failed\n", failures);
        return 1;
    }
    puts("V4 tier feedback tests: ok");
    return 0;
}
