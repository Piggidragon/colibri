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

#ifdef __cplusplus
}
#endif

#endif
