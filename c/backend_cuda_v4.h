/* Minimal CUDA backend for deepseek_v4.c.  Quantized KV and sparse MLA
 * attention move to the device; dense projections and experts remain on CPU. */
#ifndef COLIBRI_BACKEND_CUDA_V4_H
#define COLIBRI_BACKEND_CUDA_V4_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int v4_cuda_init(int device);
void v4_cuda_shutdown(void);
size_t v4_cuda_free_bytes(void);

void *v4_cuda_kv_alloc(size_t bytes);
void v4_cuda_kv_free(void *base);
int v4_cuda_kv_write_row(void *base, int slot, const void *row,
                         size_t row_bytes);

#ifdef __cplusplus
}
#endif

#endif
