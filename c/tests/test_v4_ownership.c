/* Ownership / lifecycle regression tests for DeepSeek V4 engine + session.
 * Built with -DCOLI_V4_TEST_HOOKS against objects under build/ownership/. */
#include "../deepseek_v4_internal.h"
#include "../compat.h"
#include "v4_engine_fixture.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Stubs for symbols referenced by RUNTIME but unused in these tests. */
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

static int test_index_closed_on_expert_fail(void) {
    char directory[128], error[256];
    if (v4_fixture_make(directory, sizeof(directory), "colibri-v4-own-XXXXXX")) return 1;

    coli_v4_test_fail_expert_store_open = 1;
    coli_v4_test_skip_expert_store_open = 0;
    coli_v4_test_closed_owned_index = 0;

    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions options = {.target_model_dir = directory};
    int rc = coli_v4_engine_open(&engine, &options, error, sizeof(error));
    coli_v4_test_fail_expert_store_open = 0;
    if (rc == 0 || engine) {
        fprintf(stderr, "expected expert-open failure, got success\n");
        coli_v4_engine_destroy(engine);
        v4_fixture_cleanup(directory);
        return 1;
    }
    if (coli_v4_test_closed_owned_index != 1) {
        fprintf(stderr, "expected owned index close once, got %d\n",
                coli_v4_test_closed_owned_index);
        v4_fixture_cleanup(directory);
        return 1;
    }
    v4_fixture_cleanup(directory);
    puts("ownership: index closed after expert-open failure: ok");
    return 0;
}

static int test_engine_owns_model_path(void) {
    char directory[128], error[256];
    if (v4_fixture_make(directory, sizeof(directory), "colibri-v4-own-XXXXXX")) return 1;

    char *path = strdup(directory);
    if (!path) return 1;

    coli_v4_test_fail_expert_store_open = 0;
    coli_v4_test_skip_expert_store_open = 1;
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions options = {.target_model_dir = path};
    if (coli_v4_engine_open(&engine, &options, error, sizeof(error))) {
        fprintf(stderr, "engine open failed: %s\n", error);
        free(path);
        v4_fixture_cleanup(directory);
        return 1;
    }
    free(path);

    const char *owned = coli_v4_engine_target_model_dir(engine);
    if (!owned || strcmp(owned, directory) != 0) {
        fprintf(stderr, "engine lost model path after caller free\n");
        coli_v4_engine_destroy(engine);
        v4_fixture_cleanup(directory);
        return 1;
    }
    if (!coli_v4_engine_config(engine) ||
        coli_v4_engine_config(engine)->hidden_size != 128) {
        fprintf(stderr, "engine config unusable after caller free\n");
        coli_v4_engine_destroy(engine);
        v4_fixture_cleanup(directory);
        return 1;
    }
    coli_v4_engine_destroy(engine);
    coli_v4_test_skip_expert_store_open = 0;
    v4_fixture_cleanup(directory);
    puts("ownership: engine copies model path: ok");
    return 0;
}

static int test_session_lifetime_accounting(void) {
    char directory[128], error[256];
    if (v4_fixture_make(directory, sizeof(directory), "colibri-v4-own-XXXXXX")) return 1;

    coli_v4_test_skip_expert_store_open = 1;
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions options = {.target_model_dir = directory};
    if (coli_v4_engine_open(&engine, &options, error, sizeof(error))) {
        fprintf(stderr, "engine open failed: %s\n", error);
        v4_fixture_cleanup(directory);
        return 1;
    }

    ColiV4Session *session = coli_v4_test_session_bare_create(engine);
    if (!session) {
        fprintf(stderr, "bare session create failed\n");
        coli_v4_engine_destroy(engine);
        v4_fixture_cleanup(directory);
        return 1;
    }
    if (engine->active_sessions != 1) {
        fprintf(stderr, "expected active_sessions=1 after attach, got %d\n",
                engine->active_sessions);
        coli_v4_test_session_bare_destroy(session);
        coli_v4_engine_destroy(engine);
        v4_fixture_cleanup(directory);
        return 1;
    }

    coli_v4_test_session_bare_destroy(session);
    if (engine->active_sessions != 0) {
        fprintf(stderr, "expected active_sessions=0 after detach, got %d\n",
                engine->active_sessions);
        coli_v4_engine_destroy(engine);
        v4_fixture_cleanup(directory);
        return 1;
    }
    coli_v4_engine_destroy(engine);
    coli_v4_test_skip_expert_store_open = 0;
    v4_fixture_cleanup(directory);
    puts("ownership: session lifetime accounting: ok");
    return 0;
}

static int test_vram_request_falls_back_without_cuda(void) {
    char directory[128], error[256];
    if (v4_fixture_make(directory, sizeof(directory), "colibri-v4-own-XXXXXX")) return 1;

    setenv("V4_VRAM", "1", 1);
    coli_v4_test_skip_expert_store_open = 1;
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions options = {.target_model_dir = directory};
    int result = coli_v4_engine_open(&engine, &options, error, sizeof(error));
    unsetenv("V4_VRAM");
    coli_v4_test_skip_expert_store_open = 0;
    if (result || !engine || engine->runtime.vram_enabled) {
        fprintf(stderr, "V4_VRAM did not fall back in a CPU-only build\n");
        coli_v4_engine_destroy(engine);
        v4_fixture_cleanup(directory);
        return 1;
    }
    coli_v4_engine_destroy(engine);
    v4_fixture_cleanup(directory);
    puts("ownership: V4_VRAM CPU fallback: ok");
    return 0;
}

static int test_session_tokenizer_freed(void) {
    char directory[128];
    if (v4_fixture_make(directory, sizeof(directory), "colibri-v4-own-XXXXXX")) return 1;

    char tokenizer_path[512];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             directory);

    for (int round = 0; round < 2; round++) {
        /* Mirror coli_v4_session_create / coli_v4_session_destroy tokenizer
         * ownership without pulling the full attention/runtime link set. */
        ColiV4Session session;
        memset(&session, 0, sizeof(session));
        tok_load(&session.tokenizer, tokenizer_path);
        session.tokenizer_ready = 1;

        if (!session.tokenizer_ready || !session.tokenizer.json_root ||
            session.tokenizer.n_ids < 4 || !session.tokenizer.merges.e ||
            tok_id_of(&session.tokenizer, "<eos>") != 3 ||
            hm_get(&session.tokenizer.vocab, "ab", 2) != 2) {
            fprintf(stderr, "tokenizer not owned after load\n");
            v4_fixture_cleanup(directory);
            return 1;
        }

        if (session.tokenizer_ready) {
            tok_free(&session.tokenizer);
            session.tokenizer_ready = 0;
        }
        if (session.tokenizer.json_root || session.tokenizer.merges.e ||
            session.tokenizer.vocab.e || session.tokenizer.id2str) {
            fprintf(stderr, "tokenizer not cleared after tok_free\n");
            v4_fixture_cleanup(directory);
            return 1;
        }
    }

    v4_fixture_cleanup(directory);
    puts("ownership: session tokenizer freed across create/destroy: ok");
    return 0;
}

int main(void) {
    if (test_index_closed_on_expert_fail()) return 1;
    if (test_engine_owns_model_path()) return 1;
    if (test_session_lifetime_accounting()) return 1;
    if (test_vram_request_falls_back_without_cuda()) return 1;
    if (test_session_tokenizer_freed()) return 1;
    puts("DeepSeek-V4 ownership tests: ok");
    return 0;
}
