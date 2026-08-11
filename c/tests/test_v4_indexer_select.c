/* plans/12-expert-cache-policy.md Commit 2: coli_v4_indexer_select() must
 * produce the identical set and order a full
 * qsort(scores, count, sizeof(*scores), descending_score) would put in
 * scores[0..topk) -- router semantics, not just "similar enough". This
 * compares the two against reference qsort output across the sizes and
 * tie patterns the plan calls out. coli_v4_indexer_select() itself is
 * sequential (no OpenMP) and this file has no deepseek_v4.c dependency at
 * all -- it exercises the static-inline selection algorithm from
 * deepseek_v4_internal.h in isolation. The OpenMP-parallelized scoring loop
 * that feeds it lives in coli_v4_indexer_step() in deepseek_v4.c and is not
 * covered by a direct unit test; its only regression coverage today is the
 * token-identity check in `make deepseek-v4-tiny-check` (the fixture has a
 * rate-4 CSA layer with index_topk=2, so the parallel path does run). */
#include "../deepseek_v4_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int reference_descending(const void *left, const void *right) {
    const ColiV4IndexScore *a = left, *b = right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return a->index - b->index;
}

/* xorshift32: fast, deterministic, no libc rand() state to seed/reset. */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *state = x;
}

static ColiV4IndexScore *make_scores(int count, int distinct_scores,
                                     uint32_t *rng) {
    ColiV4IndexScore *scores = malloc((size_t)count * sizeof(*scores));
    for (int i = 0; i < count; i++) {
        /* distinct_scores small (e.g. 4) forces heavy tie pressure, so the
         * ascending-index tiebreak actually gets exercised. */
        float score = distinct_scores > 0
            ? (float)(xorshift32(rng) % (uint32_t)distinct_scores)
            : (float)xorshift32(rng) / (float)UINT32_MAX;
        scores[i] = (ColiV4IndexScore){score, i};
    }
    return scores;
}

static int check_case(const char *label, int count, int topk,
                      int distinct_scores, uint32_t seed) {
    uint32_t rng = seed;
    ColiV4IndexScore *input = make_scores(count, distinct_scores, &rng);
    ColiV4IndexScore *reference = malloc((size_t)count * sizeof(*reference));
    memcpy(reference, input, (size_t)count * sizeof(*reference));
    qsort(reference, (size_t)count, sizeof(*reference), reference_descending);
    int expected_selected = topk < count ? topk : count;
    if (expected_selected < 0) expected_selected = 0;

    int got = coli_v4_indexer_select(input, count, topk);
    int failed = 0;
    if (got != expected_selected) {
        fprintf(stderr, "%s: expected selected=%d, got %d\n",
                label, expected_selected, got);
        failed = 1;
    }
    for (int i = 0; !failed && i < expected_selected; i++) {
        if (input[i].index != reference[i].index ||
            input[i].score != reference[i].score) {
            fprintf(stderr,
                    "%s: index %d differs: got {score=%g index=%d}, "
                    "want {score=%g index=%d}\n",
                    label, i, (double)input[i].score, input[i].index,
                    (double)reference[i].score, reference[i].index);
            failed = 1;
        }
    }
    free(input);
    free(reference);
    return failed;
}

static int run_matrix(const char *thread_label) {
    int failed = 0;
    static const int counts[] = {1, 512, 513, 250000};
    static const int distinct_options[] = {0, 4, 1};
    /* 0 = all-distinct scores, 4 = heavy ties, 1 = every score identical
     * (every candidate ties -- the pure index-order case). */
    for (size_t c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
        int count = counts[c];
        int topk_values[] = {1, count / 2, count, count + 37};
        for (size_t t = 0; t < sizeof(topk_values) / sizeof(*topk_values); t++) {
            int topk = topk_values[t];
            if (topk < 1 && count > 0) topk = 1;
            for (size_t d = 0; d < sizeof(distinct_options) / sizeof(*distinct_options); d++) {
                char label[128];
                snprintf(label, sizeof(label),
                         "%s count=%d topk=%d distinct=%d", thread_label,
                         count, topk, distinct_options[d]);
                failed |= check_case(label, count, topk, distinct_options[d],
                                     (uint32_t)(count * 2654435761u +
                                                (uint32_t)topk * 40503u +
                                                (uint32_t)distinct_options[d] + 1));
            }
        }
    }
    /* count == 0 edge case, not covered by the loop above. */
    failed |= check_case("count=0 topk=8", 0, 8, 0, 12345);
    /* topk <= 0 must be a clean no-op, not a crash. */
    {
        ColiV4IndexScore probe[3] = {{1.0f, 0}, {2.0f, 1}, {3.0f, 2}};
        int got = coli_v4_indexer_select(probe, 3, 0);
        if (got != 0) {
            fprintf(stderr, "%s: topk=0 expected 0 selected, got %d\n",
                    thread_label, got);
            failed = 1;
        }
        got = coli_v4_indexer_select(probe, 3, -5);
        if (got != 0) {
            fprintf(stderr, "%s: topk<0 expected 0 selected, got %d\n",
                    thread_label, got);
            failed = 1;
        }
    }
    return failed;
}

int main(void) {
    int failed = run_matrix("select");
    if (!failed) puts("test_v4_indexer_select: ok");
    return failed ? 1 : 0;
}
