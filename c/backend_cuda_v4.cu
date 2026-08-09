#include <cuda_runtime.h>

#include "backend_cuda_v4.h"
#include "backend_cuda.h"
#include "turbo_quant.h"
#include "v4_kv_codec.h"

static int g_v4_cuda_device = -1;
static int g_v4_cuda_users = 0;
static int g_v4_tables_ready = 0;

__constant__ float d_v4_centroids2[4];
__constant__ float d_v4_centroids3[8];
__constant__ float d_v4_centroids4[16];
__constant__ float d_v4_s1[COLI_TQ_GROUP];
__constant__ float d_v4_s2[COLI_TQ_GROUP];

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
    g_v4_tables_ready = 0;
}

extern "C" int v4_cuda_ready(void) {
    return g_v4_cuda_device >= 0;
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

extern "C" int v4_cuda_publish_tables(void) {
    if (g_v4_cuda_device < 0 || !coli_cuda_pipe_sync(g_v4_cuda_device))
        return -1;
    if (cudaMemcpyToSymbol(d_v4_centroids2, coli_tq_centroids2,
                           sizeof(coli_tq_centroids2)) != cudaSuccess ||
        cudaMemcpyToSymbol(d_v4_centroids3, coli_tq_centroids3,
                           sizeof(coli_tq_centroids3)) != cudaSuccess ||
        cudaMemcpyToSymbol(d_v4_centroids4, coli_tq_centroids4,
                           sizeof(coli_tq_centroids4)) != cudaSuccess ||
        cudaMemcpyToSymbol(d_v4_s1, coli_tq_s1,
                           sizeof(coli_tq_s1)) != cudaSuccess ||
        cudaMemcpyToSymbol(d_v4_s2, coli_tq_s2,
                           sizeof(coli_tq_s2)) != cudaSuccess)
        return -1;
    g_v4_tables_ready = 1;
    return 0;
}

__device__ static float v4_bits_float(unsigned int bits) {
    return __uint_as_float(bits);
}

__device__ static float v4_bf16_decode(unsigned short value) {
    return v4_bits_float((unsigned int)value << 16);
}

__device__ static float v4_bf16_round(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return v4_bits_float(bits & 0xffff0000u);
}

__device__ static float v4_fp16_decode(unsigned short value) {
    unsigned int sign = (unsigned int)(value >> 15) << 31;
    unsigned int exponent = (value >> 10) & 0x1fu;
    unsigned int mantissa = value & 0x3ffu;
    unsigned int bits;
    if (!exponent) {
        if (!mantissa) bits = sign;
        else {
            int shift = 0;
            while (!(mantissa & 0x400u)) { mantissa <<= 1; shift++; }
            bits = sign | (unsigned int)(127 - 14 - shift) << 23 |
                   (mantissa & 0x3ffu) << 13;
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + 112u) << 23 | mantissa << 13;
    }
    return v4_bits_float(bits);
}

__device__ static float v4_e8m0_decode(unsigned char value) {
    if (value == 0xffu) return NAN;
    return v4_bits_float(value ? (unsigned int)value << 23 : 0x00400000u);
}

__device__ static float v4_e4m3_decode(unsigned char value) {
    unsigned int sign = (unsigned int)(value & 0x80u) << 24;
    int exponent = (value >> 3) & 15;
    int mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    unsigned int bits = sign;
    if (exponent) {
        bits |= (unsigned int)(exponent + 120) << 23;
        bits |= (unsigned int)mantissa << 20;
    } else if (mantissa) {
        int leading = mantissa >= 4 ? 2 : mantissa >= 2 ? 1 : 0;
        bits |= (unsigned int)(leading + 118) << 23;
        bits |= (unsigned int)(mantissa - (1 << leading)) << (23 - leading);
    }
    return v4_bits_float(bits);
}

template<int Codec>
__device__ static void v4_decode_row(
    float *values, const unsigned char *row, int head_dim, int native_nope,
    size_t row_bytes, int *decode_failed) {
    for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
        if constexpr (Codec == COLI_V4_KV_F32) {
            values[column] = reinterpret_cast<const float *>(row)[column];
        } else if constexpr (Codec == COLI_V4_KV_NATIVE) {
            if (column < native_nope) {
                float scale = v4_e8m0_decode(
                    row[native_nope + column / 64]);
                values[column] = v4_bf16_round(
                    v4_e4m3_decode(row[column]) * scale);
            } else {
                int scales = (native_nope + 63) / 64;
                int offset = native_nope + scales +
                             (column - native_nope) * 2;
                unsigned short bits = (unsigned short)row[offset] |
                                      (unsigned short)row[offset + 1] << 8;
                values[column] = v4_bf16_decode(bits);
            }
        } else {
            int group = column / COLI_TQ_GROUP;
            int within = column % COLI_TQ_GROUP;
            int block_bytes = (int)(row_bytes / (head_dim / COLI_TQ_GROUP));
            const unsigned char *block = row + (size_t)group * block_bytes;
            unsigned short norm_bits = (unsigned short)block[0] |
                                       (unsigned short)block[1] << 8;
            float norm = v4_fp16_decode(norm_bits);
            if (!isfinite(norm)) {
                atomicExch(decode_failed, 1);
                values[column] = 0.0f;
                continue;
            }
            int index;
            if constexpr (Codec == COLI_V4_KV_TURBO2) {
                index = (block[2 + within / 4] >> ((within % 4) * 2)) & 3;
                values[column] = d_v4_centroids2[index] * norm;
            } else if constexpr (Codec == COLI_V4_KV_TURBO3) {
                int low = (block[2 + within / 4] >>
                           ((within % 4) * 2)) & 3;
                int high = (block[34 + within / 8] >> (within % 8)) & 1;
                values[column] = d_v4_centroids3[low | (high << 2)] * norm;
            } else {
                index = (block[2 + within / 2] >> ((within % 2) * 4)) & 15;
                values[column] = d_v4_centroids4[index] * norm;
            }
        }
    }
    __syncthreads();
    if constexpr (Codec >= COLI_V4_KV_TURBO4) {
        for (int column = threadIdx.x; column < head_dim; column += blockDim.x)
            values[column] *= d_v4_s2[column % COLI_TQ_GROUP];
        __syncthreads();
        int groups = head_dim / COLI_TQ_GROUP;
        for (int width = 1; width < COLI_TQ_GROUP; width *= 2) {
            for (int pair = threadIdx.x;
                 pair < groups * (COLI_TQ_GROUP / 2);
                 pair += blockDim.x) {
                int group = pair / (COLI_TQ_GROUP / 2);
                int local = pair % (COLI_TQ_GROUP / 2);
                int base = group * COLI_TQ_GROUP +
                    (local / width) * width * 2 + local % width;
                float left = values[base], right = values[base + width];
                values[base] = left + right;
                values[base + width] = left - right;
            }
            __syncthreads();
        }
        for (int column = threadIdx.x; column < head_dim; column += blockDim.x)
            values[column] *= 0.08838834764831845f *
                              d_v4_s1[column % COLI_TQ_GROUP];
        __syncthreads();
    }
}

template<int Codec>
__global__ static void v4_flash_attention_kernel(
    float *ctx, const float *queries,
    const unsigned char *window_kv, int window_size,
    const int *window_indices,
    const unsigned char *compressed_kv,
    const int *compressed_indices, int compressed_selected,
    const float *sinks, int head_dim, int native_nope,
    size_t row_bytes, float scale, int *decode_failed) {
    int head = blockIdx.x;
    extern __shared__ float shared[];
    float *query = shared;
    float *values = query + head_dim;
    float *accumulator = values + head_dim;
    float *reduction = accumulator + head_dim;
    __shared__ float running_max, running_sum, correction, probability;
    __shared__ int decode_failed_latch;

    for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
        query[column] = queries[(size_t)head * head_dim + column];
        accumulator[column] = 0.0f;
    }
    if (!threadIdx.x) {
        running_max = sinks[head];
        running_sum = 1.0f;
    }
    __syncthreads();

    int selected_total = window_size + compressed_selected;
    for (int rank = 0; rank < selected_total; rank++) {
        int index;
        const unsigned char *row;
        if (rank < window_size) {
            index = window_indices[rank];
            row = index < 0 ? nullptr : window_kv + (size_t)index * row_bytes;
        } else {
            int selected = rank - window_size;
            index = compressed_indices ? compressed_indices[selected] : selected;
            row = index < 0 ? nullptr
                            : compressed_kv + (size_t)index * row_bytes;
        }
        if (!row) continue;
        v4_decode_row<Codec>(values, row, head_dim, native_nope, row_bytes,
                             decode_failed);
        /* decode_failed is global and any block may set it, so a direct read
         * would let threads of this block disagree and split the barriers
         * below. Latch it once per block and branch uniformly instead. */
        if (!threadIdx.x)
            decode_failed_latch = *(volatile int *)decode_failed;
        __syncthreads();
        if (decode_failed_latch) return;
        float partial = 0.0f;
        for (int column = threadIdx.x; column < head_dim; column += blockDim.x)
            partial += query[column] * values[column];
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset; offset >>= 1) {
            if (threadIdx.x < offset)
                reduction[threadIdx.x] += reduction[threadIdx.x + offset];
            __syncthreads();
        }
        if (!threadIdx.x) {
            float score = reduction[0] * scale;
            correction = score > running_max
                ? expf(running_max - score) : 1.0f;
            if (score > running_max) {
                running_sum *= correction;
                running_max = score;
            }
            probability = expf(score - running_max);
            running_sum += probability;
            probability = v4_bf16_round(probability);
        }
        __syncthreads();
        for (int column = threadIdx.x; column < head_dim; column += blockDim.x)
            accumulator[column] = accumulator[column] * correction +
                                  probability * values[column];
        __syncthreads();
    }
    for (int column = threadIdx.x; column < head_dim; column += blockDim.x)
        ctx[(size_t)head * head_dim + column] =
            v4_bf16_round(accumulator[column] / running_sum);
}

template<int Codec>
static int v4_launch_flash(
    float *ctx, const float *q,
    const void *window_kv, int window_size, const int *window_indices,
    const void *compressed_kv,
    const int *compressed_indices, int compressed_selected,
    const float *sinks, int heads, int head_dim, int native_nope,
    size_t row_bytes, float scale) {
    size_t vector_bytes = (size_t)heads * head_dim * sizeof(float);
    float *device_q = coli_cuda_pipe_scratch(g_v4_cuda_device, 0, vector_bytes);
    float *device_ctx = coli_cuda_pipe_scratch(g_v4_cuda_device, 1, vector_bytes);
    int *device_window = reinterpret_cast<int *>(coli_cuda_pipe_scratch(
        g_v4_cuda_device, 2, (size_t)window_size * sizeof(int)));
    int *device_compressed = compressed_selected && compressed_indices
        ? reinterpret_cast<int *>(
        coli_cuda_pipe_scratch(g_v4_cuda_device, 3,
                               (size_t)compressed_selected * sizeof(int)))
        : nullptr;
    float *device_sinks = coli_cuda_pipe_scratch(
        g_v4_cuda_device, 4, (size_t)heads * sizeof(float));
    int *device_failed = reinterpret_cast<int *>(coli_cuda_pipe_scratch(
        g_v4_cuda_device, 5, sizeof(int)));
    int decode_failed = 0;
    if (!device_q || !device_ctx || !device_window || !device_sinks ||
        !device_failed ||
        (compressed_selected && compressed_indices && !device_compressed) ||
        !coli_cuda_pipe_upload(g_v4_cuda_device, device_q, q, vector_bytes) ||
        !coli_cuda_pipe_upload(g_v4_cuda_device, device_window, window_indices,
                               (size_t)window_size * sizeof(int)) ||
        !coli_cuda_pipe_upload(g_v4_cuda_device, device_sinks, sinks,
                               (size_t)heads * sizeof(float)) ||
        !coli_cuda_pipe_upload(g_v4_cuda_device, device_failed,
                               &decode_failed, sizeof(decode_failed)) ||
        (compressed_selected && compressed_indices && !coli_cuda_pipe_upload(
            g_v4_cuda_device, device_compressed, compressed_indices,
            (size_t)compressed_selected * sizeof(int))))
        return -1;
    size_t shared_bytes = ((size_t)head_dim * 3 + 256) * sizeof(float);
    v4_flash_attention_kernel<Codec><<<heads, 256, shared_bytes>>>(
        device_ctx, device_q,
        static_cast<const unsigned char *>(window_kv), window_size,
        device_window,
        static_cast<const unsigned char *>(compressed_kv),
        device_compressed, compressed_selected, device_sinks,
        head_dim, native_nope, row_bytes, scale, device_failed);
    if (cudaGetLastError() != cudaSuccess ||
        !coli_cuda_pipe_download(
            g_v4_cuda_device, device_ctx, ctx, vector_bytes) ||
        !coli_cuda_pipe_download(g_v4_cuda_device, device_failed,
                                 &decode_failed, sizeof(decode_failed)) ||
        !coli_cuda_pipe_sync(g_v4_cuda_device))
        return -1;
    return decode_failed ? -1 : 0;
}

extern "C" int v4_cuda_flash_attention(
    float *ctx, const float *q,
    const void *window_kv, int window_size, const int *window_indices,
    const void *compressed_kv, int compressed_count,
    const int *compressed_indices, int compressed_selected,
    const float *sinks, int codec, int heads, int head_dim, int rope_dim,
    size_t row_bytes, float scale) {
    if (g_v4_cuda_device < 0 || !g_v4_tables_ready || !ctx || !q ||
        !window_kv || !window_indices || !sinks || window_size < 1 ||
        compressed_count < 0 || compressed_selected < 0 ||
        (compressed_selected && !compressed_kv) ||
        heads < 1 || head_dim < 1 || head_dim > 512 || rope_dim < 0 ||
        rope_dim > head_dim || !row_bytes ||
        !(scale > 0.0f) || codec < COLI_V4_KV_F32 ||
        codec > COLI_V4_KV_TURBO2)
        return -1;
    int seen = 0;
    for (int rank = 0; rank < window_size; rank++) {
        int index = window_indices[rank];
        if (index >= window_size) return -1;
        if (index >= 0) seen++;
    }
    for (int rank = 0; rank < compressed_selected; rank++) {
        int index = compressed_indices ? compressed_indices[rank] : rank;
        if (index >= compressed_count) return -1;
        if (index >= 0) seen++;
    }
    if (!seen) return -1;

    int native_nope = 0;
    size_t expected = 0;
    if (codec == COLI_V4_KV_F32) {
        expected = (size_t)head_dim * sizeof(float);
    } else if (codec == COLI_V4_KV_NATIVE) {
        native_nope = head_dim - rope_dim;
        expected = (size_t)native_nope +
                   (size_t)(native_nope + 63) / 64 +
                   (size_t)rope_dim * 2;
    } else {
        if (head_dim % COLI_TQ_GROUP) return -1;
        int bits = codec == COLI_V4_KV_TURBO2 ? 2
                 : codec == COLI_V4_KV_TURBO3 ? 3 : 4;
        expected = (size_t)(head_dim / COLI_TQ_GROUP) *
                   coli_tq_block_bytes(bits);
    }
    if (!expected || expected != row_bytes) return -1;
    switch (codec) {
        case COLI_V4_KV_F32:
            return v4_launch_flash<COLI_V4_KV_F32>(
                ctx, q, window_kv, window_size, window_indices,
                compressed_kv, compressed_indices,
                compressed_selected, sinks, heads, head_dim, native_nope,
                row_bytes, scale);
        case COLI_V4_KV_NATIVE:
            return v4_launch_flash<COLI_V4_KV_NATIVE>(
                ctx, q, window_kv, window_size, window_indices,
                compressed_kv, compressed_indices,
                compressed_selected, sinks, heads, head_dim, native_nope,
                row_bytes, scale);
        case COLI_V4_KV_TURBO4:
            return v4_launch_flash<COLI_V4_KV_TURBO4>(
                ctx, q, window_kv, window_size, window_indices,
                compressed_kv, compressed_indices,
                compressed_selected, sinks, heads, head_dim, native_nope,
                row_bytes, scale);
        case COLI_V4_KV_TURBO3:
            return v4_launch_flash<COLI_V4_KV_TURBO3>(
                ctx, q, window_kv, window_size, window_indices,
                compressed_kv, compressed_indices,
                compressed_selected, sinks, heads, head_dim, native_nope,
                row_bytes, scale);
        case COLI_V4_KV_TURBO2:
            return v4_launch_flash<COLI_V4_KV_TURBO2>(
                ctx, q, window_kv, window_size, window_indices,
                compressed_kv, compressed_indices,
                compressed_selected, sinks, heads, head_dim, native_nope,
                row_bytes, scale);
        default: return -1;
    }
}
