#include "backend_cuda_v4.h"
#include "backend_cuda.h"

static int g_v4_cuda_device = -1;

extern "C" int v4_cuda_init(int device) {
    if (g_v4_cuda_device >= 0) return g_v4_cuda_device == device ? 0 : -1;
    if (coli_cuda_init(&device, 1) != 1) return -1;
    g_v4_cuda_device = device;
    return 0;
}

extern "C" void v4_cuda_shutdown(void) {
    if (g_v4_cuda_device < 0) return;
    coli_cuda_shutdown();
    g_v4_cuda_device = -1;
}

extern "C" size_t v4_cuda_free_bytes(void) {
    size_t free_bytes = 0, total_bytes = 0;
    if (g_v4_cuda_device < 0 ||
        !coli_cuda_mem_info(g_v4_cuda_device, &free_bytes, &total_bytes))
        return 0;
    return free_bytes;
}
