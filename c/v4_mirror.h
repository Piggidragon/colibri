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
    /* Incremented from the persistent loader thread and from ad-hoc lookups
     * on the caller's thread at once, with state->mutex released around the
     * actual read -- plain += would lose updates. */
    __atomic_fetch_add(&index->v4_mirror_bytes[replica], bytes, __ATOMIC_RELAXED);
    __atomic_fetch_add(&index->v4_mirror_reads[replica], UINT64_C(1), __ATOMIC_RELAXED);
}

static int v4_pread_loop(int fd, void *destination, size_t length, uint64_t offset) {
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
    return 0;
}

/* Same opt-out as coli_st_streaming_direct_available (deepseek_v4.c): the two
 * checks must stay in sync, but coli_st_streaming_direct_available isn't
 * declared yet at this point in the amalgam, so the one-line check is
 * duplicated here rather than forward-declared. */
static int v4_mirror_direct_enabled(void) {
    const char *setting = getenv("COLI_V4_DIRECT");
    return !(setting && atoi(setting) == 0);
}

/* Reads [offset, offset+length) for one replica through its O_DIRECT twin via
 * a private aligned bounce buffer, then copies the exact range into
 * `destination`. A bounce buffer -- not a window read straight into the
 * caller's buffer -- is required here: v4_mirror_read_striped runs one of
 * these per replica concurrently under OpenMP, and every stripe shares the
 * same nonzero alignment padding whenever the tensor's file offset isn't
 * itself 4096-aligned, so a window read would overscan `length` and race
 * with the neighboring stripe's write. Returns -1 (destination untouched) if
 * no O_DIRECT twin is available for this replica; the caller falls back to a
 * buffered pread. */
static int v4_mirror_direct_window(shards *index, int shard, int replica,
                                   uint64_t offset, size_t length,
                                   unsigned char *destination) {
    int direct_fd = v4_mirror_fd(index, shard, replica, 1);
    int buffered_fd = v4_mirror_fd(index, shard, replica, 0);
    if (direct_fd < 0 || direct_fd == buffered_fd) return -1;
    const uint64_t alignment = 4096;
    uint64_t base = offset & ~(alignment - 1);
    size_t pad = (size_t)(offset - base);
    if (length > SIZE_MAX - pad) return -1;
    size_t wanted = pad + length;
    if (wanted > SIZE_MAX - (alignment - 1)) return -1;
    size_t total = (wanted + alignment - 1) & ~(size_t)(alignment - 1);
    uint64_t file_bytes = (uint64_t)index->sizes[shard];
    if (base > file_bytes) return -1;
    uint64_t available = file_bytes - base;
    size_t direct_total = total;
    if ((uint64_t)direct_total > available)
        direct_total = (size_t)(available & ~(alignment - 1));
    unsigned char *bounce = NULL;
    if (posix_memalign((void **)&bounce, (size_t)alignment, total) || !bounce) return -1;
    int failed = 0;
    if (direct_total && v4_pread_loop(direct_fd, bounce, direct_total, base)) failed = 1;
    if (!failed && direct_total < wanted && v4_pread_loop(
            buffered_fd, bounce + direct_total, wanted - direct_total,
            base + direct_total)) failed = 1;
    if (!failed) memcpy(destination, bounce + pad, length);
    free(bounce);
    return failed ? -1 : 0;
}

static int v4_mirror_read_at(shards *index, int shard, int replica,
                             uint64_t offset, size_t length, void *destination) {
    if (!index || !destination || shard < 0 || shard >= index->nfd ||
        index->sizes[shard] < 0 || offset > (uint64_t)index->sizes[shard] ||
        length > (uint64_t)index->sizes[shard] - offset)
        return -1;
    int used = v4_mirror_replica(index, shard, replica);
    if (length && v4_mirror_direct_enabled() &&
        !v4_mirror_direct_window(index, shard, used, offset, length, destination)) {
        v4_mirror_account(index, used, length);
        return 0;
    }
    int fd = v4_mirror_fd(index, shard, used, 0);
    if (fd < 0 || v4_pread_loop(fd, destination, length, offset)) return -1;
    v4_mirror_account(index, used, length);
    return 0;
}

static int v4_mirror_prefetch_at(shards *index, int shard, int replica,
                                 uint64_t offset, size_t length) {
    if (!index || shard < 0 || shard >= index->nfd ||
        index->sizes[shard] < 0 || offset > (uint64_t)index->sizes[shard] ||
        length > (uint64_t)index->sizes[shard] - offset)
        return -1;
    int used = v4_mirror_replica(index, shard, replica);
    int fd = v4_mirror_fd(index, shard, used, 0);
    return fd < 0 ? -1 : posix_fadvise(fd, (off_t)offset, (off_t)length,
                                        POSIX_FADV_WILLNEED);
}

/* A full replica is required for striping: partial mirrors retain the routed
 * single-read path. v4_mirror_prefetch_striped below computes identical
 * bounds so a prefetch and its later demand read always name the same set of
 * fds, stripe for stripe. */
static int v4_mirror_stripe_ready(const shards *index, int shard, size_t length) {
    if (!index || !index->v4_mirror_active || length < (size_t)(4 << 20)) return 0;
    int replicas = index->v4_mirror_reps;
    for (int i = 1; i < replicas; i++)
        if (v4_mirror_replica(index, shard, i) != i) return 0;
    return 1;
}

static int v4_mirror_stripe_bounds(const shards *index, int replicas,
                                   size_t length, size_t bound[V4_MIRROR_REPS + 1]) {
    bound[0] = 0;
    size_t previous = 0;
    for (int i = 0; i < replicas; i++) {
        int cut = index->v4_mirror_cut[i];
        size_t end = i + 1 == replicas ? length :
            ((length * (size_t)cut / 256u + 4095u) & ~(size_t)4095u);
        if (end > length) end = length;
        if (end < previous) return -1;
        bound[i + 1] = end; previous = end;
    }
    return 0;
}

static int v4_mirror_read_striped(shards *index, int shard, uint64_t offset,
                                  size_t length, void *destination) {
    if (!v4_mirror_stripe_ready(index, shard, length)) return -1;
    int replicas = index->v4_mirror_reps;
    size_t bound[V4_MIRROR_REPS + 1] = {0};
    if (v4_mirror_stripe_bounds(index, replicas, length, bound)) return -1;
    int failed = 0;
    #pragma omp parallel for schedule(static, 1) num_threads(replicas) reduction(|:failed)
    for (int i = 0; i < replicas; i++) {
        size_t begin = bound[i], bytes = bound[i + 1] - begin;
        if (bytes && v4_mirror_read_at(index, shard, i, offset + begin, bytes,
                                       (unsigned char *)destination + begin)) failed = 1;
    }
    return failed ? -1 : 0;
}

/* Mirrors v4_mirror_read_striped's bounds exactly so a prefetch warms the
 * same per-drive ranges the later demand read (v4_mirror_read_striped) pulls
 * from -- see the eligibility comment on v4_mirror_stripe_ready above. */
static int v4_mirror_prefetch_striped(shards *index, int shard, uint64_t offset,
                                      size_t length) {
    if (!v4_mirror_stripe_ready(index, shard, length)) return -1;
    int replicas = index->v4_mirror_reps;
    size_t bound[V4_MIRROR_REPS + 1] = {0};
    if (v4_mirror_stripe_bounds(index, replicas, length, bound)) return -1;
    int failed = 0;
    for (int i = 0; i < replicas; i++) {
        size_t begin = bound[i], bytes = bound[i + 1] - begin;
        if (bytes && v4_mirror_prefetch_at(index, shard, i, offset + begin, bytes))
            failed = 1;
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
