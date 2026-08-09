#include "backend_cuda_v4.h"
#include "backend_cuda.h"

static int g_v4_cuda_device = -1;
static int g_v4_cuda_users = 0;

extern "C" int v4_cuda_init(int device) {
    if (g_v4_cuda_device >= 0) {
        if (g_v4_cuda_device != device) return -1;
        g_v4_cuda_users++;
        return 0;
    }
    if (coli_cuda_init(&device, 1) != 1) return -1;
    g_v4_cuda_device = device;
    g_v4_cuda_users = 1;
    return 0;
}

extern "C" void v4_cuda_shutdown(void) {
    if (g_v4_cuda_device < 0) return;
    if (--g_v4_cuda_users > 0) return;
    coli_cuda_shutdown();
    g_v4_cuda_device = -1;
    g_v4_cuda_users = 0;
}

extern "C" size_t v4_cuda_free_bytes(void) {
    size_t free_bytes = 0, total_bytes = 0;
    if (g_v4_cuda_device < 0 ||
        !coli_cuda_mem_info(g_v4_cuda_device, &free_bytes, &total_bytes))
        return 0;
    return free_bytes;
}

extern "C" void *v4_cuda_kv_alloc(size_t bytes) {
    if (g_v4_cuda_device < 0 || !bytes) return nullptr;
    return coli_cuda_pipe_alloc(g_v4_cuda_device, bytes);
}

extern "C" void v4_cuda_kv_free(void *base) {
    if (g_v4_cuda_device >= 0 && base)
        coli_cuda_pipe_free(g_v4_cuda_device, base);
}

extern "C" int v4_cuda_kv_write_row(void *base, int slot, const void *row,
                                      size_t row_bytes) {
    if (g_v4_cuda_device < 0 || !base || slot < 0 || !row || !row_bytes ||
        (size_t)slot > SIZE_MAX / row_bytes)
        return -1;
    unsigned char *destination = static_cast<unsigned char *>(base) +
                                 (size_t)slot * row_bytes;
    return coli_cuda_pipe_upload(
        g_v4_cuda_device, destination, row, row_bytes) ? 0 : -1;
}
