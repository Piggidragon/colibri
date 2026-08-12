/* omp_tune.h — dimensionamento della squadra OpenMP sui CORE FISICI.
 *
 * PERCHE' SOLO IL DIMENSIONAMENTO, E NON LO SPIN-WAIT.
 * Il blocco di tuning in colibri.c fa due cose che hanno profili di rischio
 * OPPOSTI, e vanno tenute separate:
 *
 *   dimensionamento  OMP_NUM_THREADS = core fisici (niente SMT)
 *                    #718: +2.3x su Zen3 (5950X, 16C/32T) solo cambiando
 *                    il numero di thread. Il guadagno e' cosi' grande che
 *                    "sommerge la maggior parte dei delta che si citano qui".
 *
 *   spin-wait        OMP_WAIT_POLICY=active, GOMP_SPINCOUNT, KMP_BLOCKTIME
 *                    #707: -2.2x sul decode di un host a bassa residenza
 *                          (M1 Max 32 GB, ~10% di expert residenti)
 *                    #116: -39% su Metal      #159: ~3x su x86+CUDA
 *                    #341: 3000% di CPU su FreeBSD con la squadra ferma
 *                    Meccanismo: dove il token e' fatto di byte dal disco,
 *                    una squadra che gira a vuoto ruba i core al pool di I/O
 *                    che sta facendo il lavoro vero.
 *
 * Kimi K3 e OLMoE non avevano NESSUNA delle due. Qui prendono solo la prima:
 * Kimi e' il motore piu' disk-bound del progetto (misurato: 6.7% di hit,
 * 891 GB letti per 32 token), cioe' esattamente il regime in cui la seconda
 * meta' fa danno. Aggiungergliela sarebbe stato un peggioramento misurabile.
 * GLM now calls this same helper after its optional hot-team re-exec, so it
 * receives the physical-core sizing without changing the independent spin-wait
 * policy that its existing block controls.
 *
 * PERCHE' QUI NON SERVE IL RE-EXEC.
 * colibri.c si ri-esegue perche' OMP_WAIT_POLICY & co. le legge il COSTRUTTORE
 * di libgomp, prima di main(): un setenv() dentro main() arriva tardi. Il numero
 * di thread no: omp_set_num_threads() e' una API di runtime e ha effetto subito.
 * La meta' sicura e' anche la meta' semplice.
 *
 * REGOLA SUI FALLIMENTI: se il conteggio dei core fisici non e' determinabile,
 * NON si indovina — si lascia il default di OpenMP. Un conteggio sbagliato e'
 * peggio di nessun conteggio (cfr. #325, dove un fallback silenzioso a 1
 * inchiodava il decode su un core solo).
 */
#ifndef COLI_OMP_TUNE_H
#define COLI_OMP_TUNE_H

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <dirent.h>
#endif

/* Numero di core FISICI, o 0 se non determinabile. Mai un valore inventato. */
#if defined(_WIN32)
static int coli_count_windows_physical_cores(const void *buf, DWORD bytes)
{
    const char *p = (const char *)buf;
    const char *end = p + bytes;
    const size_t header_size = offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                                        Processor);
    int cores = 0;

    while ((size_t)(end - p) >= header_size) {
        LOGICAL_PROCESSOR_RELATIONSHIP relationship;
        DWORD record_size;
        memcpy(&relationship,
               p + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Relationship),
               sizeof(relationship));
        memcpy(&record_size,
               p + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Size),
               sizeof(record_size));
        if (record_size < header_size || (size_t)(end - p) < record_size)
            break; /* Reject a zero, truncated, or otherwise malformed record. */
        if (relationship == RelationProcessorCore) cores++;
        p += record_size;
    }
    return cores;
}
#endif

static int coli_physical_cores(void)
{
#if defined(_WIN32)
    DWORD need = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &need);
    if (!need) return 0;
    void *buf = malloc(need);
    if (!buf) return 0;
    int cores = GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &need)
                    ? coli_count_windows_physical_cores(buf, need)
                    : 0;
    free(buf);
    return cores;

#elif defined(__APPLE__)
    /* hw.perflevel0.logicalcpu = i core PERFORMANCE. Su Apple Silicon
     * hw.physicalcpu li conta tutti, E-core compresi (10 su un M1 Max), e con
     * una barriera per matmul e' il thread piu' lento a dettare il passo: gli
     * E-core rallentano la squadra invece di aiutarla (#707, -4.2% decode).
     * Su Intel Mac perflevel* non esiste: li' hw.physicalcpu e' corretto. */
    int v = 0; size_t sz = sizeof(v);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    v = 0; sz = sizeof(v);
    if (sysctlbyname("hw.physicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    return 0;

#else
    /* Linux: un core fisico = una lista di thread_siblings distinta. Contare le
     * liste uniche deduplica l'SMT senza dover interpretare la topologia. */
    DIR *d = opendir("/sys/devices/system/cpu");
    if (!d) return 0;
    char seen[1024][64];
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) && n < 1024) {
        if (strncmp(e->d_name, "cpu", 3) != 0 || e->d_name[3] < '0' || e->d_name[3] > '9')
            continue;
        /* d_name can be up to 255 bytes; leave enough room for the fixed
         * sysfs prefix/suffix so -Wformat-truncation stays honest when this
         * shared helper is compiled into the GLM engine too. */
        char path[512], line[64];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/%s/topology/thread_siblings_list", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        if (fgets(line, sizeof(line), f)) {
            line[strcspn(line, "\n")] = 0;
            int dup = 0;
            for (int i = 0; i < n; i++) if (strcmp(seen[i], line) == 0) { dup = 1; break; }
            if (!dup) { snprintf(seen[n], sizeof(seen[0]), "%s", line); n++; }
        }
        fclose(f);
    }
    closedir(d);
    return n;
#endif
}

#if !defined(_WIN32) && !defined(__APPLE__)
/* cpumap prints the most significant group first; count nibbles from its
 * right edge so CPU 0 is the least-significant bit regardless of commas. */
static int coli_linux_cpumap_has_cpu(const char *map, int cpu)
{
    if (!map || cpu < 0) return 0;
    int wanted = cpu / 4, nibble = 0;
    const char *p = map + strlen(map);
    while (p > map) {
        unsigned char c = (unsigned char)*--p;
        if (c == ',' || c == '\n' || c == ' ' || c == '\t') continue;
        int value = c >= '0' && c <= '9' ? c - '0' :
                    c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                    c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (value < 0) return 0;
        if (nibble++ == wanted) return value & (1 << (cpu % 4));
    }
    return 0;
}

/* Intel hybrid kernels export the P-core logical CPUs here.  Collapse their
 * SMT siblings as coli_physical_cores does, because static V4 matmuls want
 * six P-core workers on the target i5-13400F, not twelve SMT threads. */
static int coli_linux_performance_cores(void)
{
    DIR *types = opendir("/sys/devices/system/cpu/types");
    if (!types) return 0;
    char cpu_map[1024] = {0};
    struct dirent *type;
    while ((type = readdir(types))) {
        if (strncmp(type->d_name, "intel_core_", 11) != 0) continue;
        char path[512];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/types/%s/cpumap", type->d_name);
        FILE *f = fopen(path, "r");
        if (f) {
            fgets(cpu_map, sizeof(cpu_map), f);
            fclose(f);
        }
        if (cpu_map[0]) break;
    }
    closedir(types);
    if (!cpu_map[0]) return 0;

    DIR *cpus = opendir("/sys/devices/system/cpu");
    if (!cpus) return 0;
    char seen[1024][64];
    int n = 0;
    struct dirent *entry;
    while ((entry = readdir(cpus)) && n < 1024) {
        if (strncmp(entry->d_name, "cpu", 3) != 0 ||
            entry->d_name[3] < '0' || entry->d_name[3] > '9') continue;
        int cpu = atoi(entry->d_name + 3);
        if (!coli_linux_cpumap_has_cpu(cpu_map, cpu)) continue;
        char path[512], siblings[64];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/%s/topology/thread_siblings_list",
                 entry->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        if (fgets(siblings, sizeof(siblings), f)) {
            siblings[strcspn(siblings, "\n")] = 0;
            int duplicate = 0;
            for (int i = 0; i < n; i++)
                if (strcmp(seen[i], siblings) == 0) { duplicate = 1; break; }
            if (!duplicate) {
                snprintf(seen[n], sizeof(seen[0]), "%s", siblings);
                n++;
            }
        }
        fclose(f);
    }
    closedir(cpus);
    return n;
}
#endif

/* Dimensiona la squadra OpenMP sui core fisici. Rispetta OMP_NUM_THREADS se
 * l'utente l'ha impostata, e non fa nulla se il conteggio non e' affidabile.
 * `engine` finisce solo nella riga di log. */
static void coli_omp_tune_threads(const char *engine)
{
#ifdef _OPENMP
    const char *off = getenv("COLI_NO_OMP_TUNE");
    if (off) return;                       /* stesso kill-switch degli altri motori */
    if (getenv("OMP_NUM_THREADS")) return; /* l'utente comanda */

    int phys = coli_physical_cores();
    if (phys <= 0) return;                 /* sconosciuto -> default di OpenMP */
    int logical = omp_get_max_threads();
    if (phys >= logical) return;           /* niente SMT da evitare: silenzio */

    omp_set_num_threads(phys);
    fprintf(stderr, "[OMP] %s: %d physical-core threads instead of %d logical CPUs; "
                    "SMT can halve decode throughput on some CPUs (#718); "
                    "set OMP_NUM_THREADS=<n> to override\n",
            engine, phys, logical);
#else
    (void)engine;
#endif
}

/* V4's static matmuls are barrier-bound on the target's mixed P/E topology.
 * Keep this engine-local: the other engines retain their established generic
 * physical-core policy.  OMP_NUM_THREADS remains the explicit, universal
 * override; V4_OMP_CORES only decides the otherwise automatic team. */
static void coli_v4_omp_tune_threads(const char *engine)
{
#ifdef _OPENMP
    const char *off = getenv("COLI_NO_OMP_TUNE");
    if (off || getenv("OMP_NUM_THREADS")) return;

    int logical = omp_get_max_threads();
    if (logical < 1) return;
    const char *setting = getenv("V4_OMP_CORES");
    int all = !setting || !setting[0] || !strcmp(setting, "all");
    int use_perf = setting && !strcmp(setting, "perf");
    int target = 0, performance = 0;
#if !defined(_WIN32) && !defined(__APPLE__)
    if (use_perf) {
        target = coli_linux_performance_cores();
        performance = target > 0;
    }
#endif
    if (!target && (all || use_perf))
        target = coli_physical_cores();
    if (!all && !use_perf) {
        char *end = NULL;
        long parsed = strtol(setting, &end, 10);
        if (end != setting && !*end) {
            if (parsed < 1) target = 1;
            else if (parsed > logical) target = logical;
            else target = (int)parsed;
        } else {
#if !defined(_WIN32) && !defined(__APPLE__)
            target = coli_linux_performance_cores();
            performance = target > 0;
#endif
            if (!target) target = coli_physical_cores();
        }
    }
    if (target <= 0 || target >= logical) return;

    omp_set_num_threads(target);
    fprintf(stderr, "[OMP] %s: %d %s-core threads instead of %d logical CPUs; "
                    "set V4_OMP_CORES=perf|all|<n> or OMP_NUM_THREADS=<n> to override\n",
            engine, target, performance ? "performance" :
            (all || use_perf ? "physical" : "selected"),
            logical);
#else
    (void)engine;
#endif
}

#endif /* COLI_OMP_TUNE_H */
