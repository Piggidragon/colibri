/* Minimal on-disk V4 checkpoint for tests that need a real coli_v4_engine_open.
 * Shared by the ownership and tiering tests; each caller passes its own mkdtemp
 * prefix so two suites can run side by side in the same working directory. */
#ifndef COLIBRI_TESTS_V4_ENGINE_FIXTURE_H
#define COLIBRI_TESTS_V4_ENGINE_FIXTURE_H

#include "../compat.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int v4_fixture_write_all(int fd, const void *data, size_t length) {
    const unsigned char *bytes = (const unsigned char *)data;
    while (length) {
        ssize_t count = write(fd, bytes, length);
        if (count <= 0) return -1;
        bytes += count;
        length -= (size_t)count;
    }
    return 0;
}

static int v4_fixture_write_config(const char *directory) {
    static const char json[] =
        "{\"model_type\":\"deepseek_v4\",\"expert_dtype\":\"fp4\","
        "\"scoring_func\":\"sqrtsoftplus\",\"topk_method\":\"noaux_tc\","
        "\"hidden_size\":128,\"num_hidden_layers\":1,"
        "\"num_attention_heads\":4,\"head_dim\":32,\"q_lora_rank\":64,"
        "\"qk_rope_head_dim\":8,\"o_groups\":2,\"o_lora_rank\":64,"
        "\"sliding_window\":16,\"index_n_heads\":4,\"index_head_dim\":16,"
        "\"index_topk\":8,\"n_routed_experts\":8,\"num_experts_per_tok\":2,"
        "\"n_shared_experts\":1,\"moe_intermediate_size\":32,"
        "\"num_hash_layers\":1,\"num_nextn_predict_layers\":1,"
        "\"hc_mult\":4,\"hc_sinkhorn_iters\":5,\"vocab_size\":256,"
        "\"max_position_embeddings\":4096,\"rms_norm_eps\":1e-6,"
        "\"hc_eps\":1e-6,\"routed_scaling_factor\":1.5,\"swiglu_limit\":10,"
        "\"rope_theta\":10000,\"compress_rope_theta\":40000,"
        "\"compress_ratios\":[0,0],"
        "\"rope_scaling\":{\"original_max_position_embeddings\":1024,"
        "\"beta_fast\":32,\"beta_slow\":1,\"factor\":4},"
        "\"quantization_config\":{\"fmt\":\"e4m3\",\"scale_fmt\":\"ue8m0\"}}";
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", directory);
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) return -1;
    int result = v4_fixture_write_all(fd, json, sizeof(json) - 1);
    close(fd);
    return result;
}

static int v4_fixture_write_safetensors(const char *directory) {
    static const char header[] =
        "{\"dense.fp8\":{\"dtype\":\"F8_E4M3\",\"shape\":[2,2],"
        "\"data_offsets\":[0,4]},"
        "\"expert.fp4\":{\"dtype\":\"I8\",\"shape\":[2,1],"
        "\"data_offsets\":[4,6]},"
        "\"expert.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[2,1],"
        "\"data_offsets\":[6,8]}}";
    static const unsigned char payload[] = {1, 2, 3, 4, 5, 6, 7, 8};
    char path[512];
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors",
             directory);
    uint64_t header_length = sizeof(header) - 1;
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) return -1;
    int result = v4_fixture_write_all(fd, &header_length,
                                      sizeof(header_length)) ||
                 v4_fixture_write_all(fd, header, (size_t)header_length) ||
                 v4_fixture_write_all(fd, payload, sizeof(payload));
    close(fd);
    return result;
}

static int v4_fixture_write_tokenizer(const char *directory) {
    static const char json[] =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"ab\":2},"
        "\"merges\":[\"a b\"]},"
        "\"added_tokens\":[{\"id\":3,\"content\":\"<eos>\",\"special\":true}]}";
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", directory);
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) return -1;
    int result = v4_fixture_write_all(fd, json, sizeof(json) - 1);
    close(fd);
    return result;
}

/* `prefix` must end in exactly six X for mkdtemp.  Native MinGW binaries do not
 * resolve the MSYS /tmp mount, so the directory is created relative to cwd. */
static int v4_fixture_make(char *directory, size_t directory_size,
                           const char *prefix) {
    char scratch[128];
    if (strlen(prefix) + 1 > sizeof(scratch)) return -1;
    memcpy(scratch, prefix, strlen(prefix) + 1);
    if (!mkdtemp(scratch)) return -1;
    if (strlen(scratch) + 1 > directory_size) return -1;
    memcpy(directory, scratch, strlen(scratch) + 1);
    if (v4_fixture_write_config(directory) ||
        v4_fixture_write_safetensors(directory) ||
        v4_fixture_write_tokenizer(directory))
        return -1;
    return 0;
}

static void v4_fixture_cleanup(const char *directory) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", directory);
    unlink(path);
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors",
             directory);
    unlink(path);
    snprintf(path, sizeof(path), "%s/tokenizer.json", directory);
    unlink(path);
    rmdir(directory);
}

#endif
