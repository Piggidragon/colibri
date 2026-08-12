/* V4's optional model mirror.  This intentionally lives beside the V4 index
 * rather than refactoring colibri.c's mature GLM path: Plan 11 removes that
 * engine, while the V4 reader must keep deterministic replica selection. */
#ifndef COLIBRI_V4_MIRROR_H
#define COLIBRI_V4_MIRROR_H

#include "st.h"

#define V4_MIRROR_REPS (1 + ST_MAX_MIR)

static inline int v4_mirror_enabled(const shards *index) {
    return index && index->v4_mirror_active;
}

static int v4_mirror_route(const shards *index, int layer, int expert) {
    if (!index || !index->v4_mirror_active) return 0;
    uint32_t hash = (uint32_t)layer * UINT32_C(2654435761) ^
        (uint32_t)expert * UINT32_C(0x9e3779b9);
    hash ^= hash >> 16; hash *= UINT32_C(0x45d9f3b); hash ^= hash >> 16;
    int replica = 0, value = (int)(hash & 255);
    while (value >= index->v4_mirror_cut[replica]) replica++;
    return replica;
}

static int v4_mirror_fd(const shards *index, int shard, int replica, int direct) {
    if (!index || shard < 0 || shard >= index->nfd) return -1;
    int fd = direct ? st_direct_fd_rep((shards *)index, index->fds[shard], replica)
                    : st_fd_rep((shards *)index, index->fds[shard], replica);
    return fd >= 0 ? fd : (direct ? index->dfds[shard] : index->fds[shard]);
}

static int v4_mirror_replica(const shards *index, int shard, int requested) {
    if (!index || requested <= 0 || requested > index->nrep || shard < 0 ||
        shard >= index->nfd || st_fd_rep((shards *)index, index->fds[shard], requested) < 0)
        return 0;
    return requested;
}

static void v4_mirror_account(shards *index, int replica, uint64_t bytes) {
    if (!index || replica < 0 || replica >= V4_MIRROR_REPS) return;
    index->v4_mirror_bytes[replica] += bytes;
    index->v4_mirror_reads[replica]++;
}

static int v4_mirror_read_at(shards *index, int shard, int replica,
                             uint64_t offset, size_t length, void *destination) {
    int used = v4_mirror_replica(index, shard, replica);
    int fd = v4_mirror_fd(index, shard, used, 0);
    if (fd < 0) return -1;
    unsigned char *output = destination;
    size_t done = 0;
    while (done < length) {
        ssize_t count = pread(fd, output + done, length - done,
                              (off_t)(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return -1;
        done += (size_t)count;
    }
    v4_mirror_account(index, used, length);
    return 0;
}

static int v4_mirror_prefetch_at(shards *index, int shard, int replica,
                                 uint64_t offset, size_t length) {
    int used = v4_mirror_replica(index, shard, replica);
    int fd = v4_mirror_fd(index, shard, used, 0);
    return fd < 0 ? -1 : posix_fadvise(fd, (off_t)offset, (off_t)length,
                                        POSIX_FADV_WILLNEED);
}

/* A full replica is required for striping: partial mirrors retain the routed
 * single-read path, so a prefetch and its demand read always name one fd. */
static int v4_mirror_read_striped(shards *index, int shard, uint64_t offset,
                                  size_t length, void *destination) {
    if (!index || !index->v4_mirror_active || length < (size_t)(4 << 20)) return -1;
    int replicas = index->v4_mirror_reps;
    for (int i = 1; i < replicas; i++)
        if (v4_mirror_replica(index, shard, i) != i) return -1;
    size_t bound[V4_MIRROR_REPS + 1] = {0};
    size_t previous = 0;
    for (int i = 0; i < replicas; i++) {
        int cut = index->v4_mirror_cut[i];
        size_t end = i + 1 == replicas ? length :
            ((length * (size_t)cut / 256u + 4095u) & ~(size_t)4095u);
        if (end > length) end = length;
        if (end < previous) return -1;
        bound[i + 1] = end; previous = end;
    }
    int failed = 0;
    #pragma omp parallel for schedule(static, 1) reduction(|:failed)
    for (int i = 0; i < replicas; i++) {
        size_t begin = bound[i], bytes = bound[i + 1] - begin;
        if (bytes && v4_mirror_read_at(index, shard, i, offset + begin, bytes,
                                       (unsigned char *)destination + begin)) failed = 1;
    }
    return failed ? -1 : 0;
}

static int v4_mirror_setup(shards *index) {
    const char *setting = getenv("COLI_MODEL_MIRROR");
    if (!index || !setting || !*setting) return 0;
    st_mirror_reset(index);
    memset(index->v4_mirror_bytes, 0, sizeof(index->v4_mirror_bytes));
    memset(index->v4_mirror_reads, 0, sizeof(index->v4_mirror_reads));
    index->v4_mirror_active = 0; index->v4_mirror_reps = 1;
    char directories[4096];
    snprintf(directories, sizeof(directories), "%s", setting);
    char *cursor = directories;
    while (cursor && *cursor && index->nrep < ST_MAX_MIR) {
        char *end = cursor;
        while (*end && *end != ';' && *end != ',') end++;
        int last = !*end; *end = 0;
        while (*cursor == ' ') cursor++;
        size_t length = strlen(cursor);
        while (length && cursor[length - 1] == ' ') cursor[--length] = 0;
        if (*cursor) {
            int files = st_mirror_add(index, cursor);
            if (files > 0)
                fprintf(stderr, "v4_mirror dir=%s shards=%d/%d replica=%d\n",
                        cursor, files, index->nfd, index->nrep);
            else
                fprintf(stderr, "v4_mirror warning=no-usable-shards dir=%s\n", cursor);
        }
        cursor = last ? NULL : end + 1;
    }
    if (index->nrep < 1) {
        fprintf(stderr, "v4_mirror disabled=no-usable-replica\n");
        return 0;
    }
    double weight[V4_MIRROR_REPS] = {0};
    int replicas = index->nrep + 1, valid = 1, count = 0;
    const char *weights = getenv("COLI_DISK_WEIGHTS");
    if (weights && *weights) {
        char copy[256]; snprintf(copy, sizeof(copy), "%s", weights);
        for (char *item = strtok(copy, ", "); item; item = strtok(NULL, ", ")) {
            char *tail = NULL; double value = strtod(item, &tail);
            if (!tail || *tail || value <= 0 || count >= replicas) { valid = 0; break; }
            weight[count++] = value;
        }
        if (count != replicas) valid = 0;
    } else valid = 0;
    if (!valid) {
        for (int i = 0; i < replicas; i++) weight[i] = 1.0;
        if (weights && *weights)
            fprintf(stderr, "v4_mirror warning=invalid-weights value=%s fallback=equal\n",
                    weights);
    }
    double total = 0; for (int i = 0; i < replicas; i++) total += weight[i];
    double cumulative = 0; int previous = 0;
    for (int i = 0; i < replicas; i++) {
        cumulative += weight[i];
        int cut = (int)(256.0 * cumulative / total + 0.5);
        if (cut <= previous) cut = previous + 1;
        if (cut > 256) cut = 256;
        index->v4_mirror_cut[i] = cut; previous = cut;
    }
    index->v4_mirror_cut[replicas - 1] = 256;
    index->v4_mirror_reps = replicas; index->v4_mirror_active = 1;
    fprintf(stderr, "v4_mirror drives=%d split", replicas);
    for (int i = 0; i < replicas; i++) {
        int low = i ? index->v4_mirror_cut[i - 1] : 0;
        fprintf(stderr, "%s %.0f%%", i ? "/" : "", 100.0 *
                (index->v4_mirror_cut[i] - low) / 256.0);
    }
    fprintf(stderr, " source=%s\n", valid ? "COLI_DISK_WEIGHTS" : "equal-fallback");
    return 1;
}

static void v4_mirror_report(const shards *index, FILE *stream) {
    if (!index || !index->v4_mirror_active || !stream) return;
    fprintf(stream, "MIRROR: primary %.2f GB (%llu reads)",
            index->v4_mirror_bytes[0] / 1e9,
            (unsigned long long)index->v4_mirror_reads[0]);
    for (int i = 1; i < index->v4_mirror_reps; i++)
        fprintf(stream, " | mirror%d %.2f GB (%llu reads)", i,
                index->v4_mirror_bytes[i] / 1e9,
                (unsigned long long)index->v4_mirror_reads[i]);
    fputc('\n', stream);
}

#endif
