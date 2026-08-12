/* V4 mirror routing is deliberately separate from colibri.c's GLM globals.
 * Exercise the no-mirror fast path, weighted deterministic routing and a
 * replica read against tiny byte-identical safetensors files. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../v4_mirror.h"

#include <sys/stat.h>
#include <unistd.h>

#define PRIMARY "tmp_v4_mirror_primary"
#define MIRROR "tmp_v4_mirror_replica"

static void cleanup(void);

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        cleanup(); \
        return 1; \
    } \
} while (0)

static int write_shard(const char *directory) {
    char path[256];
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    FILE *stream = fopen(path, "wb");
    if (!stream) return -1;
    enum { PAYLOAD = 4 * 1024 * 1024 };
    const char header[] =
        "{\"t0\":{\"dtype\":\"U8\",\"shape\":[4194304],\"data_offsets\":[0,4194304]}}";
    uint64_t length = sizeof(header) - 1;
    unsigned char *payload = malloc(PAYLOAD);
    if (!payload) { fclose(stream); return -1; }
    for (int i = 0; i < PAYLOAD; i++) payload[i] = (unsigned char)(i * 17 + 3);
    int result = fwrite(&length, sizeof(length), 1, stream) != 1 ||
        fwrite(header, 1, (size_t)length, stream) != length ||
        fwrite(payload, 1, PAYLOAD, stream) != PAYLOAD;
    free(payload);
    fclose(stream);
    return result ? -1 : 0;
}

static void cleanup(void) {
    char path[256];
    const char *dirs[] = {PRIMARY, MIRROR};
    for (size_t i = 0; i < sizeof(dirs) / sizeof(dirs[0]); i++) {
        snprintf(path, sizeof(path), "%s/model.safetensors", dirs[i]);
        remove(path); rmdir(dirs[i]);
    }
}

int main(void) {
    cleanup(); unsetenv("COLI_MODEL_MIRROR"); unsetenv("COLI_DISK_WEIGHTS");
    CHECK(mkdir(PRIMARY, 0700) == 0 && mkdir(MIRROR, 0700) == 0);
    CHECK(write_shard(PRIMARY) == 0 && write_shard(MIRROR) == 0);

    shards index;
    st_init(&index, PRIMARY);
    CHECK(v4_mirror_setup(&index) == 0);
    CHECK(!v4_mirror_enabled(&index) && index.nrep == 0);

    CHECK(setenv("COLI_MODEL_MIRROR", MIRROR, 1) == 0);
    CHECK(setenv("COLI_DISK_WEIGHTS", "2,1", 1) == 0);
    CHECK(v4_mirror_setup(&index) == 1);
    CHECK(v4_mirror_enabled(&index) && index.v4_mirror_reps == 2);
    CHECK(index.v4_mirror_cut[0] >= 169 && index.v4_mirror_cut[0] <= 172);

    int routed[2] = {0, 0};
    for (int layer = 0; layer < 43; layer++)
        for (int expert = 0; expert < 256; expert++)
            routed[v4_mirror_route(&index, layer, expert)]++;
    CHECK(routed[0] > routed[1] && routed[0] * 10 > routed[1] * 17 &&
          routed[0] * 10 < routed[1] * 23);

    enum { PAYLOAD = 4 * 1024 * 1024 };
    unsigned char *got = malloc(PAYLOAD);
    CHECK(got != NULL);
    st_tensor *tensor = st_find(&index, "t0");
    CHECK(tensor && v4_mirror_read_at(&index, 0, 1, (uint64_t)tensor->off,
                                       PAYLOAD, got) == 0);
    for (int i = 0; i < PAYLOAD; i++) CHECK(got[i] == (unsigned char)(i * 17 + 3));
    CHECK(index.v4_mirror_bytes[1] == PAYLOAD && index.v4_mirror_reads[1] == 1);
    memset(got, 0, PAYLOAD);
    CHECK(v4_mirror_read_striped(&index, 0, (uint64_t)tensor->off, PAYLOAD, got) == 0);
    for (int i = 0; i < PAYLOAD; i++) CHECK(got[i] == (unsigned char)(i * 17 + 3));
    CHECK(index.v4_mirror_bytes[0] && index.v4_mirror_bytes[1] > PAYLOAD);
    free(got);

    st_mirror_reset(&index); unsetenv("COLI_MODEL_MIRROR"); unsetenv("COLI_DISK_WEIGHTS");
    close(index.fds[0]); if (index.dfds[0] >= 0) close(index.dfds[0]);
    free(index.paths[0]); free(index.t[0].name); free(index.t); free(index.hidx);
    cleanup();
    puts("V4 mirror tests: ok");
    return 0;
}
