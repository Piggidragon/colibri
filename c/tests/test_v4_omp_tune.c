/* V4 keeps automatic OpenMP sizing in the binary because Linux hybrid CPU
 * topology is unavailable to the generic Python launchers. */
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "../omp_tune.h"

static void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

static int expected_physical(int logical) {
    int target = coli_physical_cores();
    return target > 0 && target < logical ? target : logical;
}

static int expected_performance(int logical) {
    int target = 0;
#if !defined(_WIN32) && !defined(__APPLE__)
    target = coli_linux_performance_cores();
#endif
    if (!target) target = coli_physical_cores();
    return target > 0 && target < logical ? target : logical;
}

int main(void) {
#ifndef _OPENMP
    puts("test_v4_omp_tune: ok (OpenMP unavailable)");
    return 0;
#else
    int failed = 0;
    int logical = omp_get_num_procs();
    int sentinel = logical > 1 ? logical - 1 : 1;

    set_env("COLI_NO_OMP_TUNE", NULL);
    set_env("OMP_NUM_THREADS", NULL);
    set_env("V4_OMP_CORES", NULL);
    omp_set_num_threads(logical);
    coli_v4_omp_tune_threads("test");
    if (omp_get_max_threads() != expected_physical(logical)) failed = 1;

    set_env("V4_OMP_CORES", "perf");
    omp_set_num_threads(logical);
    coli_v4_omp_tune_threads("test");
    if (omp_get_max_threads() != expected_performance(logical)) failed = 1;

    set_env("V4_OMP_CORES", "1");
    omp_set_num_threads(logical);
    coli_v4_omp_tune_threads("test");
    if (omp_get_max_threads() != 1) failed = 1;

    set_env("V4_OMP_CORES", "garbage");
    omp_set_num_threads(logical);
    coli_v4_omp_tune_threads("test");
    if (omp_get_max_threads() != expected_physical(logical)) failed = 1;

    set_env("V4_OMP_CORES", "1");
    set_env("OMP_NUM_THREADS", "7");
    omp_set_num_threads(sentinel);
    coli_v4_omp_tune_threads("test");
    if (omp_get_max_threads() != sentinel) failed = 1;

    set_env("OMP_NUM_THREADS", NULL);
    set_env("COLI_NO_OMP_TUNE", "1");
    omp_set_num_threads(sentinel);
    coli_v4_omp_tune_threads("test");
    if (omp_get_max_threads() != sentinel) failed = 1;

    set_env("COLI_NO_OMP_TUNE", NULL);
    set_env("V4_OMP_CORES", NULL);
    if (failed) {
        fprintf(stderr, "test_v4_omp_tune: FAIL\n");
        return 1;
    }
    puts("test_v4_omp_tune: ok");
    return 0;
#endif
}
