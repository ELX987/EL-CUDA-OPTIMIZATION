/*****************************************************************************************
 * EL_ternCUDA_kernel.cu
 *
 * Native W1.58A8 BitLinear CUDA training primitive for PyTorch extension use.
 * Build compatibility patch: no the cuBLAS development header and no libcublas link dependency.
 *
 * Public layout expected by EL_ternCUDA_kernel.h:
 *   X          : __half  row-major [M, N]
 *   W_shadow   : float   row-major [K, N]     // K = output features
 *   W_packed   : uint32  row-major [K, ceil(N / 16)]
 *   W_scale    : float   [K]                  // absmean scale per output row
 *   Y          : __half  row-major [M, K]
 *   dY         : __half  row-major [M, K]
 *   dX         : __half  row-major [M, N]
 *   dW         : float   row-major [K, N]
 *
 * Ternary encoding, 16 weights per uint32:
 *   00 ->  0
 *   01 -> +1
 *   10 -> -1
 *   11 ->  0 / reserved
 * This is branchless sign/mask decoding: q = bit0 - bit1.
 *
 * Notes:
 *   - Forward path: activation FP16 -> per-row INT8, ternary W packed -> int8 lanes,
 *     __dp4a -> INT32 accumulation, dequant to FP16.
 *   - dX is a correctness-oriented low-bit kernel.
 *   - dW uses a self-contained CUDA fallback kernel; the training script defaults to
 *     PyTorch/cuBLAS for dW outside this extension so minimal CUDA images do not need
 *     cuBLAS development headers during extension compilation.
 *****************************************************************************************/
#include "EL_ternCUDA_kernel.h"

#define EL_TERNARY_CUDA_KERNEL_VERSION "2026-05-01.headerless-no-cublas.dx-prescale-v8"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifndef EL_TERNARY_PER_WORD
#define EL_TERNARY_PER_WORD 16
#endif

#ifndef EL_TERNARY_BITS
#define EL_TERNARY_BITS 2
#endif

#ifndef EL_EPSILON
#define EL_EPSILON 1.0e-8f
#endif

#ifndef EL_QUANT_THREADS
#define EL_QUANT_THREADS 256
#endif

#ifndef EL_PACK_THREADS
#define EL_PACK_THREADS 256
#endif

// Forward block computes a 16 x 32 output tile and streams the input dimension in chunks.
#ifndef EL_FWD_BLOCK_M
#define EL_FWD_BLOCK_M 16
#endif

#ifndef EL_FWD_BLOCK_K
#define EL_FWD_BLOCK_K 32
#endif

#ifndef EL_FWD_BLOCK_N
#define EL_FWD_BLOCK_N 256
#endif

// dX block computes a 16 x 16 tile of dX and streams output features in chunks.
#ifndef EL_DX_BLOCK_M
#define EL_DX_BLOCK_M 16
#endif

#ifndef EL_DX_BLOCK_N
#define EL_DX_BLOCK_N 16
#endif

#ifndef EL_DX_STEP_K
#define EL_DX_STEP_K 256
#endif

// Baseline dW block computes a 16 x 16 tile of dW and streams batch/tokens in chunks.
#ifndef EL_DW_BLOCK_K
#define EL_DW_BLOCK_K 16
#endif

#ifndef EL_DW_BLOCK_N
#define EL_DW_BLOCK_N 16
#endif

#ifndef EL_DW_STEP_M
#define EL_DW_STEP_M 128
#endif

using el_packed_t = uint32_t;
using el_i8_t = int8_t;

#define EL_RETURN_IF_CUDA_ERROR(expr)            \
    do {                                         \
        cudaError_t _err = (expr);               \
        if (_err != cudaSuccess) return _err;    \
    } while (0)

static inline int el_ceil_div_host(int a, int b) {
    return (a + b - 1) / b;
}


// --------------------------------------------------------------------------------------
// Device helpers
// --------------------------------------------------------------------------------------

__host__ __device__ __forceinline__ int el_ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ int el_pack_i8x4(el_i8_t x0, el_i8_t x1, el_i8_t x2, el_i8_t x3) {
    uint32_t u = 0u;
    u |= (uint32_t)(uint8_t)x0;
    u |= (uint32_t)(uint8_t)x1 << 8;
    u |= (uint32_t)(uint8_t)x2 << 16;
    u |= (uint32_t)(uint8_t)x3 << 24;
    return (int)u;
}

__device__ __forceinline__ el_i8_t el_ternary_code_to_i8(uint32_t code) {
    // 00 -> 0, 01 -> +1, 10 -> -1, 11 -> 0/reserved.
    const int bit0 = (int)(code & 0x1u);
    const int bit1 = (int)((code >> 1) & 0x1u);
    return (el_i8_t)(bit0 - bit1);
}

__device__ __forceinline__ uint32_t el_i8_to_ternary_code(el_i8_t q) {
    // Branchless-ish sign/mask encoding: pos bit at b0, neg bit at b1.
    const uint32_t pos = (uint32_t)(q > 0);
    const uint32_t neg = (uint32_t)(q < 0);
    return pos | (neg << 1);
}

__device__ __forceinline__ int el_unpack_ternary4_i8x4(uint32_t word, int start_in_word) {
    const uint32_t c0 = (word >> (EL_TERNARY_BITS * (start_in_word + 0))) & 0x3u;
    const uint32_t c1 = (word >> (EL_TERNARY_BITS * (start_in_word + 1))) & 0x3u;
    const uint32_t c2 = (word >> (EL_TERNARY_BITS * (start_in_word + 2))) & 0x3u;
    const uint32_t c3 = (word >> (EL_TERNARY_BITS * (start_in_word + 3))) & 0x3u;
    return el_pack_i8x4(
        el_ternary_code_to_i8(c0),
        el_ternary_code_to_i8(c1),
        el_ternary_code_to_i8(c2),
        el_ternary_code_to_i8(c3));
}

__device__ __forceinline__ el_i8_t el_decode_ternary_at(uint32_t word, int offset_in_word) {
    const uint32_t code = (word >> (EL_TERNARY_BITS * offset_in_word)) & 0x3u;
    return el_ternary_code_to_i8(code);
}

__device__ __forceinline__ int el_dp4a_s32(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    return __dp4a(a, b, c);
#else
    const int a0 = (int)(int8_t)((uint32_t)a & 0xffu);
    const int a1 = (int)(int8_t)(((uint32_t)a >> 8) & 0xffu);
    const int a2 = (int)(int8_t)(((uint32_t)a >> 16) & 0xffu);
    const int a3 = (int)(int8_t)(((uint32_t)a >> 24) & 0xffu);
    const int b0 = (int)(int8_t)((uint32_t)b & 0xffu);
    const int b1 = (int)(int8_t)(((uint32_t)b >> 8) & 0xffu);
    const int b2 = (int)(int8_t)(((uint32_t)b >> 16) & 0xffu);
    const int b3 = (int)(int8_t)(((uint32_t)b >> 24) & 0xffu);
    return c + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
#endif
}

__device__ __forceinline__ el_i8_t el_float_to_i8_symmetric(float v, float inv_scale) {
    int q = __float2int_rn(v * inv_scale);
    q = q < -127 ? -127 : q;
    q = q > 127 ? 127 : q;
    return (el_i8_t)q;
}

__device__ __forceinline__ el_i8_t el_shadow_to_ternary(float w, float absmean_scale) {
    const float thr = 0.5f * absmean_scale;
    const int pos = (w >= thr);
    const int neg = (w <= -thr);
    return (el_i8_t)(pos - neg);
}

// --------------------------------------------------------------------------------------
// Quantize activations: FP16 rows -> INT8 rows + per-row scale.
// scale_x[row] = absmax(row) / 127, xq = round(x / scale_x).
// --------------------------------------------------------------------------------------

__global__ __launch_bounds__(EL_QUANT_THREADS, 2) void el_quantize_fp16_rows_i8_kernel(
    const __half* __restrict__ x,
    el_i8_t* __restrict__ xq,
    float* __restrict__ scale,
    int rows,
    int cols) {

    const int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float s_absmax[EL_QUANT_THREADS];

    float local_absmax = 0.0f;
    const int base = row * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float v = __half2float(x[base + col]);
        local_absmax = fmaxf(local_absmax, fabsf(v));
    }

    s_absmax[threadIdx.x] = local_absmax;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_absmax[threadIdx.x] = fmaxf(s_absmax[threadIdx.x], s_absmax[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float row_scale = fmaxf(s_absmax[0] / 127.0f, EL_EPSILON);
    const float inv_scale = 1.0f / row_scale;
    if (threadIdx.x == 0) scale[row] = row_scale;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float v = __half2float(x[base + col]);
        xq[base + col] = el_float_to_i8_symmetric(v, inv_scale);
    }
}

// --------------------------------------------------------------------------------------
// Pack W_shadow[K, N] -> W_packed[K, ceil(N / 16)] and W_scale[K].
// --------------------------------------------------------------------------------------

__global__ __launch_bounds__(EL_PACK_THREADS, 2) void el_pack_ternary_rowmajor_kernel(
    const float* __restrict__ w_shadow,
    el_packed_t* __restrict__ w_packed,
    float* __restrict__ w_scale,
    int K,
    int N) {

    const int out = blockIdx.x;
    if (out >= K) return;

    __shared__ float s_sum_abs[EL_PACK_THREADS];
    __shared__ float s_scale;

    float local_sum = 0.0f;
    const int row_base = out * N;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        local_sum += fabsf(w_shadow[row_base + n]);
    }

    s_sum_abs[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum_abs[threadIdx.x] += s_sum_abs[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float scale = fmaxf(s_sum_abs[0] / (float)N, EL_EPSILON);
        s_scale = scale;
        w_scale[out] = scale;
    }
    __syncthreads();

    const int words = el_ceil_div_int(N, EL_TERNARY_PER_WORD);
    for (int word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
        uint32_t packed = 0u;
        const int base_n = word_idx * EL_TERNARY_PER_WORD;
#pragma unroll
        for (int lane = 0; lane < EL_TERNARY_PER_WORD; ++lane) {
            const int n = base_n + lane;
            el_i8_t q = 0;
            if (n < N) {
                q = el_shadow_to_ternary(w_shadow[row_base + n], s_scale);
            }
            const uint32_t code = el_i8_to_ternary_code(q);
            packed |= code << (EL_TERNARY_BITS * lane);
        }
        w_packed[out * words + word_idx] = packed;
    }
}


// --------------------------------------------------------------------------------------
// Dequantize packed ternary weights to FP16 [K, N]. This is a support path for the
// optional Tensor-Core dX backward mode: dX = dY @ dequant(W_ternary).
// --------------------------------------------------------------------------------------

__global__ __launch_bounds__(256, 2) void el_dequantize_packed_to_fp16_kernel(
    const el_packed_t* __restrict__ w_packed,
    const float* __restrict__ w_scale,
    __half* __restrict__ w_deq,
    int K,
    int N) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (k >= K || n >= N) return;
    const int words = el_ceil_div_int(N, EL_TERNARY_PER_WORD);
    const int word_idx = n >> 4;
    const int lane = n & 15;
    const uint32_t word = w_packed[k * words + word_idx];
    const int q = (int)el_decode_ternary_at(word, lane);
    const float v = (float)q * w_scale[k];
    w_deq[k * N + n] = __float2half_rn(v);
}

// --------------------------------------------------------------------------------------
// Forward GEMM, prequantized activations:
//   Y[M, K] = Xq[M, N] @ Wternary[K, N]^T, then dequantize by x_scale[m] * w_scale[k].
// Optimization vs the earlier prototype:
//   - true GEMM, not GEMV
//   - pre-pack X into shared int32 x4 lanes once per tile
//   - pre-decode ternary W into shared int32 w4 lanes once per tile
//   - use __dp4a with INT32 accumulation
// --------------------------------------------------------------------------------------

template<int BM, int BO, int BN>
__global__ __launch_bounds__(BM * BO, 1) void el_forward_w158a8_kernel(
    const el_i8_t* __restrict__ xq,
    const float* __restrict__ x_scale,
    const el_packed_t* __restrict__ w_packed,
    const float* __restrict__ w_scale,
    __half* __restrict__ y,
    int32_t* __restrict__ acc_optional,
    int M,
    int N,
    int K) {

    static_assert(BN % 16 == 0, "BN must be divisible by 16");
    static_assert(BN % 4 == 0, "BN must be divisible by 4");

    constexpr int X4_GROUPS = BN / 4;
    __shared__ int s_x4[BM * X4_GROUPS];
    __shared__ int s_w4[BO * X4_GROUPS];

    const int local_o = threadIdx.x;  // [0, BO)
    const int local_m = threadIdx.y;  // [0, BM)
    const int tid = local_m * BO + local_o;
    constexpr int THREADS = BM * BO;

    const int m = blockIdx.y * BM + local_m;
    const int out = blockIdx.x * BO + local_o;
    const int words_total = el_ceil_div_int(N, EL_TERNARY_PER_WORD);

    int32_t acc = 0;

    for (int n0 = 0; n0 < N; n0 += BN) {
        // Load/pack Xq tile into shared 4-lane int8 vectors.
        for (int idx = tid; idx < BM * X4_GROUPS; idx += THREADS) {
            const int lm = idx / X4_GROUPS;
            const int g4 = idx - lm * X4_GROUPS;
            const int gm = blockIdx.y * BM + lm;
            const int gn = n0 + g4 * 4;
            el_i8_t x0 = 0, x1 = 0, x2 = 0, x3 = 0;
            if (gm < M) {
                if (gn + 0 < N) x0 = xq[gm * N + gn + 0];
                if (gn + 1 < N) x1 = xq[gm * N + gn + 1];
                if (gn + 2 < N) x2 = xq[gm * N + gn + 2];
                if (gn + 3 < N) x3 = xq[gm * N + gn + 3];
            }
            s_x4[idx] = el_pack_i8x4(x0, x1, x2, x3);
        }

        // Decode W tile into shared 4-lane int8 vectors.
        for (int idx = tid; idx < BO * X4_GROUPS; idx += THREADS) {
            const int lo = idx / X4_GROUPS;
            const int g4 = idx - lo * X4_GROUPS;
            const int gout = blockIdx.x * BO + lo;
            const int gn = n0 + g4 * 4;
            const int word_idx = gn / EL_TERNARY_PER_WORD;
            const int offset = gn & (EL_TERNARY_PER_WORD - 1);
            uint32_t word = 0u;
            if (gout < K && word_idx < words_total) {
                word = w_packed[gout * words_total + word_idx];
            }
            s_w4[idx] = el_unpack_ternary4_i8x4(word, offset);
        }
        __syncthreads();

        if (m < M && out < K) {
#pragma unroll 4
            for (int g4 = 0; g4 < X4_GROUPS; ++g4) {
                const int x4 = s_x4[local_m * X4_GROUPS + g4];
                const int w4 = s_w4[local_o * X4_GROUPS + g4];
                acc = el_dp4a_s32(x4, w4, acc);
            }
        }
        __syncthreads();
    }

    if (m < M && out < K) {
        if (acc_optional != nullptr) acc_optional[m * K + out] = acc;
        const float yf = (float)acc * x_scale[m] * w_scale[out];
        y[m * K + out] = __float2half_rn(yf);
    }
}

// --------------------------------------------------------------------------------------
// dX kernel:
//   dX[M, N] = dY[M, K] @ dequant(Wternary[K, N])
// Scales vary by output row, so this is a float accumulation kernel.
// --------------------------------------------------------------------------------------

template<int BM, int BN, int STEP_K>
__global__ __launch_bounds__(BM * BN, 2) void el_backward_input_kernel(
    const __half* __restrict__ dy,
    const el_packed_t* __restrict__ w_packed,
    const float* __restrict__ w_scale,
    __half* __restrict__ dx,
    int M,
    int N,
    int K) {

    static_assert(BN == 16, "BN=16 matches one packed ternary word lane group");

    // Pre-scale dY by the per-output-row ternary weight scale once when staging
    // the [M, K] tile. This removes BN repeated float multiplies from the inner
    // loop, which is helpful when the packed dX path is selected.
    __shared__ __half s_dy_scaled[BM * STEP_K];
    __shared__ el_packed_t s_w_word[STEP_K];

    const int local_n = threadIdx.x;  // [0, 16)
    const int local_m = threadIdx.y;  // [0, BM)
    const int tid = local_m * BN + local_n;
    constexpr int THREADS = BM * BN;

    const int m = blockIdx.y * BM + local_m;
    const int n = blockIdx.x * BN + local_n;
    const int words_total = el_ceil_div_int(N, EL_TERNARY_PER_WORD);
    const int word_n = blockIdx.x;  // one input-feature packed word per block.x

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += STEP_K) {
        for (int idx = tid; idx < BM * STEP_K; idx += THREADS) {
            const int lm = idx / STEP_K;
            const int lk = idx - lm * STEP_K;
            const int gm = blockIdx.y * BM + lm;
            const int gk = k0 + lk;
            float v = 0.0f;
            if (gm < M && gk < K) {
                v = __half2float(dy[gm * K + gk]) * w_scale[gk];
            }
            s_dy_scaled[idx] = __float2half_rn(v);
        }

        // Load each packed ternary word once per output feature instead of 16 times.
        for (int idx = tid; idx < STEP_K; idx += THREADS) {
            const int gk = k0 + idx;
            s_w_word[idx] = (gk < K && word_n < words_total) ? w_packed[gk * words_total + word_n] : 0u;
        }
        __syncthreads();

        if (m < M && n < N) {
#pragma unroll 4
            for (int lk = 0; lk < STEP_K; ++lk) {
                const float dyv = __half2float(s_dy_scaled[local_m * STEP_K + lk]);
                const float q = (float)el_decode_ternary_at(s_w_word[lk], local_n);
                acc += dyv * q;
            }
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        dx[m * N + n] = __float2half_rn(acc);
    }
}


// Quantized dX fast path:
//   First fold W_scale into dY and quantize each token row to INT8.
//   Then compute dX = int8(dY * W_scale) @ ternary(W)^T with dp4a and INT32 accumulation.
// This path is approximate but avoids the float accumulation in el_backward_input_kernel.
__global__ __launch_bounds__(EL_QUANT_THREADS, 2) void el_quantize_scaled_dy_rows_i8_kernel(
    const __half* __restrict__ dy,
    const float* __restrict__ w_scale,
    el_i8_t* __restrict__ dy_q,
    float* __restrict__ dy_scale,
    int M,
    int K) {

    const int row = blockIdx.x;
    if (row >= M) return;

    __shared__ float s_absmax[EL_QUANT_THREADS];
    float local_absmax = 0.0f;
    const int base = row * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float v = __half2float(dy[base + k]) * w_scale[k];
        local_absmax = fmaxf(local_absmax, fabsf(v));
    }
    s_absmax[threadIdx.x] = local_absmax;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_absmax[threadIdx.x] = fmaxf(s_absmax[threadIdx.x], s_absmax[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float scale = fmaxf(s_absmax[0] / 127.0f, EL_EPSILON);
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) dy_scale[row] = scale;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float v = __half2float(dy[base + k]) * w_scale[k];
        dy_q[base + k] = el_float_to_i8_symmetric(v, inv_scale);
    }
}

template<int BM, int BN, int STEP_K>
__global__ __launch_bounds__(BM * BN, 2) void el_backward_input_quantized_kernel(
    const el_i8_t* __restrict__ dy_q,
    const float* __restrict__ dy_scale,
    const el_packed_t* __restrict__ w_packed,
    __half* __restrict__ dx,
    int M,
    int N,
    int K) {

    static_assert(BN == 16, "BN=16 matches one packed ternary word lane group");
    static_assert((STEP_K % 4) == 0, "STEP_K must be a multiple of four for dp4a");

    __shared__ el_i8_t s_dy_q[BM * STEP_K];
    __shared__ el_packed_t s_w_word[STEP_K];

    const int local_n = threadIdx.x;
    const int local_m = threadIdx.y;
    const int tid = local_m * BN + local_n;
    constexpr int THREADS = BM * BN;

    const int m = blockIdx.y * BM + local_m;
    const int n = blockIdx.x * BN + local_n;
    const int words_total = el_ceil_div_int(N, EL_TERNARY_PER_WORD);
    const int word_n = blockIdx.x;

    int acc = 0;

    for (int k0 = 0; k0 < K; k0 += STEP_K) {
        for (int idx = tid; idx < BM * STEP_K; idx += THREADS) {
            const int lm = idx / STEP_K;
            const int lk = idx - lm * STEP_K;
            const int gm = blockIdx.y * BM + lm;
            const int gk = k0 + lk;
            s_dy_q[idx] = (gm < M && gk < K) ? dy_q[gm * K + gk] : (el_i8_t)0;
        }
        for (int idx = tid; idx < STEP_K; idx += THREADS) {
            const int gk = k0 + idx;
            s_w_word[idx] = (gk < K && word_n < words_total) ? w_packed[gk * words_total + word_n] : 0u;
        }
        __syncthreads();

        if (m < M && n < N) {
#pragma unroll 4
            for (int lk = 0; lk < STEP_K; lk += 4) {
                const int a = el_pack_i8x4(
                    s_dy_q[local_m * STEP_K + lk + 0],
                    s_dy_q[local_m * STEP_K + lk + 1],
                    s_dy_q[local_m * STEP_K + lk + 2],
                    s_dy_q[local_m * STEP_K + lk + 3]);
                const int b = el_pack_i8x4(
                    el_decode_ternary_at(s_w_word[lk + 0], local_n),
                    el_decode_ternary_at(s_w_word[lk + 1], local_n),
                    el_decode_ternary_at(s_w_word[lk + 2], local_n),
                    el_decode_ternary_at(s_w_word[lk + 3], local_n));
                acc = el_dp4a_s32(a, b, acc);
            }
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        dx[m * N + n] = __float2half_rn((float)acc * dy_scale[m]);
    }
}

// --------------------------------------------------------------------------------------
// Baseline dW STE kernel:
//   dW[K, N] = dY[M, K]^T @ X[M, N]
// Correct but scalar. The Python wrapper can use torch/cuBLAS for this path when faster.
// --------------------------------------------------------------------------------------

template<int BO, int BN, int STEP_M>
__global__ void el_ste_weight_grad_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    float* __restrict__ dw,
    int M,
    int N,
    int K) {

    __shared__ __half s_x[STEP_M * BN];
    __shared__ __half s_dy[STEP_M * BO];

    const int local_n = threadIdx.x;  // input feature within BN
    const int local_o = threadIdx.y;  // output feature within BO
    const int tid = local_o * BN + local_n;
    constexpr int THREADS = BO * BN;

    const int n = blockIdx.x * BN + local_n;
    const int out = blockIdx.y * BO + local_o;

    float acc = 0.0f;

    for (int m0 = 0; m0 < M; m0 += STEP_M) {
        for (int idx = tid; idx < STEP_M * BN; idx += THREADS) {
            const int lm = idx / BN;
            const int ln = idx - lm * BN;
            const int gm = m0 + lm;
            const int gn = blockIdx.x * BN + ln;
            s_x[idx] = (gm < M && gn < N) ? x[gm * N + gn] : __float2half(0.0f);
        }

        for (int idx = tid; idx < STEP_M * BO; idx += THREADS) {
            const int lm = idx / BO;
            const int lo = idx - lm * BO;
            const int gm = m0 + lm;
            const int go = blockIdx.y * BO + lo;
            s_dy[idx] = (gm < M && go < K) ? dy[gm * K + go] : __float2half(0.0f);
        }
        __syncthreads();

        if (out < K && n < N) {
#pragma unroll 4
            for (int lm = 0; lm < STEP_M; ++lm) {
                acc += __half2float(s_dy[lm * BO + local_o]) * __half2float(s_x[lm * BN + local_n]);
            }
        }
        __syncthreads();
    }

    if (out < K && n < N) {
        dw[out * N + n] = acc;
    }
}

// --------------------------------------------------------------------------------------
// Public C ABI
// --------------------------------------------------------------------------------------

extern "C" int el_bitlinear_packed_words_per_row(int N) {
    return el_ceil_div_host(N, EL_TERNARY_PER_WORD);
}

extern "C" size_t el_bitlinear_packed_weight_bytes(int K, int N) {
    return (size_t)K * (size_t)el_bitlinear_packed_words_per_row(N) * sizeof(el_packed_t);
}

extern "C" size_t el_bitlinear_forward_temp_bytes(int M, int N) {
    return (size_t)M * (size_t)N * sizeof(el_i8_t) + (size_t)M * sizeof(float);
}

extern "C" size_t el_bitlinear_forward_from_shadow_temp_bytes(int M, int N, int K) {
    return el_bitlinear_forward_temp_bytes(M, N)
         + el_bitlinear_packed_weight_bytes(K, N)
         + (size_t)K * sizeof(float);
}

extern "C" size_t el_bitlinear_backward_from_shadow_temp_bytes(int N, int K) {
    return el_bitlinear_packed_weight_bytes(K, N) + (size_t)K * sizeof(float);
}

extern "C" cudaError_t el_pack_ternary_weights_async(
    const float* W_shadow,
    uint32_t* W_packed,
    float* W_scale,
    int K,
    int N,
    cudaStream_t stream) {

    if (W_shadow == nullptr || W_packed == nullptr || W_scale == nullptr || K <= 0 || N <= 0) {
        return cudaErrorInvalidValue;
    }

    el_pack_ternary_rowmajor_kernel<<<K, EL_PACK_THREADS, 0, stream>>>(
        W_shadow, reinterpret_cast<el_packed_t*>(W_packed), W_scale, K, N);
    return cudaGetLastError();
}

extern "C" cudaError_t el_dequantize_packed_to_fp16_async(
    const uint32_t* W_packed,
    const float* W_scale,
    __half* W_deq,
    int K,
    int N,
    cudaStream_t stream) {

    if (W_packed == nullptr || W_scale == nullptr || W_deq == nullptr || K <= 0 || N <= 0) {
        return cudaErrorInvalidValue;
    }
    dim3 block(256, 1, 1);
    dim3 grid(el_ceil_div_host(N, 256), K, 1);
    el_dequantize_packed_to_fp16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const el_packed_t*>(W_packed), W_scale, W_deq, K, N);
    return cudaGetLastError();
}

extern "C" cudaError_t el_dequantize_packed_weights_half_async(
    const uint32_t* W_packed,
    const float* W_scale,
    __half* W_deq,
    int K,
    int N,
    cudaStream_t stream) {

    return el_dequantize_packed_to_fp16_async(W_packed, W_scale, W_deq, K, N, stream);
}

extern "C" cudaError_t el_quantize_fp16_per_row_int8_async(
    const __half* X,
    int8_t* X_q,
    float* X_scale,
    int rows,
    int cols,
    cudaStream_t stream) {

    if (X == nullptr || X_q == nullptr || X_scale == nullptr || rows <= 0 || cols <= 0) {
        return cudaErrorInvalidValue;
    }

    el_quantize_fp16_rows_i8_kernel<<<rows, EL_QUANT_THREADS, 0, stream>>>(
        X, reinterpret_cast<el_i8_t*>(X_q), X_scale, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t el_bitlinear_forward_prequantized_async(
    const int8_t* X_q,
    const float* X_scale,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* Y,
    int32_t* Acc_optional,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    if (X_q == nullptr || X_scale == nullptr || W_packed == nullptr || W_scale == nullptr || Y == nullptr ||
        M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(EL_FWD_BLOCK_K, EL_FWD_BLOCK_M, 1);
    dim3 grid(el_ceil_div_host(K, EL_FWD_BLOCK_K), el_ceil_div_host(M, EL_FWD_BLOCK_M), 1);

    el_forward_w158a8_kernel<EL_FWD_BLOCK_M, EL_FWD_BLOCK_K, EL_FWD_BLOCK_N><<<grid, block, 0, stream>>>(
        reinterpret_cast<const el_i8_t*>(X_q), X_scale,
        reinterpret_cast<const el_packed_t*>(W_packed), W_scale,
        Y, Acc_optional, M, N, K);
    return cudaGetLastError();
}

extern "C" cudaError_t el_bitlinear_forward_async(
    const __half* X,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* Y,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    if (X == nullptr || W_packed == nullptr || W_scale == nullptr || Y == nullptr || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    el_i8_t* X_q = nullptr;
    float* X_scale = nullptr;
    const size_t xq_bytes = (size_t)M * (size_t)N * sizeof(el_i8_t);
    const size_t xs_bytes = (size_t)M * sizeof(float);

    cudaError_t alloc0 = cudaMallocAsync((void**)&X_q, xq_bytes, stream);
    if (alloc0 != cudaSuccess) return alloc0;
    cudaError_t alloc1 = cudaMallocAsync((void**)&X_scale, xs_bytes, stream);
    if (alloc1 != cudaSuccess) {
        cudaFreeAsync(X_q, stream);
        return alloc1;
    }

    cudaError_t err = el_quantize_fp16_per_row_int8_async(X, reinterpret_cast<int8_t*>(X_q), X_scale, M, N, stream);
    if (err == cudaSuccess) {
        err = el_bitlinear_forward_prequantized_async(
            reinterpret_cast<const int8_t*>(X_q), X_scale, W_packed, W_scale, Y, nullptr, M, N, K, stream);
    }

    cudaError_t free0 = cudaFreeAsync(X_q, stream);
    cudaError_t free1 = cudaFreeAsync(X_scale, stream);
    if (err != cudaSuccess) return err;
    if (free0 != cudaSuccess) return free0;
    if (free1 != cudaSuccess) return free1;
    return cudaSuccess;
}

extern "C" cudaError_t el_bitlinear_forward_from_shadow_async(
    const __half* X,
    const float* W_shadow,
    __half* Y,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    if (X == nullptr || W_shadow == nullptr || Y == nullptr || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    const int words = el_bitlinear_packed_words_per_row(N);
    el_packed_t* W_packed = nullptr;
    float* W_scale = nullptr;
    const size_t wp_bytes = (size_t)K * (size_t)words * sizeof(el_packed_t);
    const size_t ws_bytes = (size_t)K * sizeof(float);

    cudaError_t alloc0 = cudaMallocAsync((void**)&W_packed, wp_bytes, stream);
    if (alloc0 != cudaSuccess) return alloc0;
    cudaError_t alloc1 = cudaMallocAsync((void**)&W_scale, ws_bytes, stream);
    if (alloc1 != cudaSuccess) {
        cudaFreeAsync(W_packed, stream);
        return alloc1;
    }

    cudaError_t err = el_pack_ternary_weights_async(
        W_shadow, reinterpret_cast<uint32_t*>(W_packed), W_scale, K, N, stream);
    if (err == cudaSuccess) {
        err = el_bitlinear_forward_async(X, reinterpret_cast<const uint32_t*>(W_packed), W_scale, Y, M, N, K, stream);
    }

    cudaError_t free0 = cudaFreeAsync(W_packed, stream);
    cudaError_t free1 = cudaFreeAsync(W_scale, stream);
    if (err != cudaSuccess) return err;
    if (free0 != cudaSuccess) return free0;
    if (free1 != cudaSuccess) return free1;
    return cudaSuccess;
}

extern "C" cudaError_t el_bitlinear_backward_input_async(
    const __half* dY,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* dX,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    if (dY == nullptr || W_packed == nullptr || W_scale == nullptr || dX == nullptr || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(EL_DX_BLOCK_N, EL_DX_BLOCK_M, 1);
    dim3 grid(el_ceil_div_host(N, EL_DX_BLOCK_N), el_ceil_div_host(M, EL_DX_BLOCK_M), 1);

    el_backward_input_kernel<EL_DX_BLOCK_M, EL_DX_BLOCK_N, EL_DX_STEP_K><<<grid, block, 0, stream>>>(
        dY, reinterpret_cast<const el_packed_t*>(W_packed), W_scale, dX, M, N, K);
    return cudaGetLastError();
}


extern "C" cudaError_t el_bitlinear_backward_input_quantized_async(
    const __half* dY,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* dX,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    if (dY == nullptr || W_packed == nullptr || W_scale == nullptr || dX == nullptr || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    el_i8_t* dY_q = nullptr;
    float* dY_scale = nullptr;
    const size_t q_bytes = (size_t)M * (size_t)K * sizeof(el_i8_t);
    const size_t s_bytes = (size_t)M * sizeof(float);
    cudaError_t err = cudaMallocAsync((void**)&dY_q, q_bytes, stream);
    if (err != cudaSuccess) return err;
    err = cudaMallocAsync((void**)&dY_scale, s_bytes, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(dY_q, stream);
        return err;
    }

    el_quantize_scaled_dy_rows_i8_kernel<<<M, EL_QUANT_THREADS, 0, stream>>>(
        dY, W_scale, dY_q, dY_scale, M, K);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        dim3 block(EL_DX_BLOCK_N, EL_DX_BLOCK_M, 1);
        dim3 grid(el_ceil_div_host(N, EL_DX_BLOCK_N), el_ceil_div_host(M, EL_DX_BLOCK_M), 1);
        el_backward_input_quantized_kernel<EL_DX_BLOCK_M, EL_DX_BLOCK_N, EL_DX_STEP_K><<<grid, block, 0, stream>>>(
            dY_q, dY_scale, reinterpret_cast<const el_packed_t*>(W_packed), dX, M, N, K);
        err = cudaGetLastError();
    }

    cudaError_t free0 = cudaFreeAsync(dY_q, stream);
    cudaError_t free1 = cudaFreeAsync(dY_scale, stream);
    if (err != cudaSuccess) return err;
    if (free0 != cudaSuccess) return free0;
    if (free1 != cudaSuccess) return free1;
    return cudaSuccess;
}



extern "C" cudaError_t el_bitlinear_ste_weight_grad_async(
    const __half* X,
    const __half* dY,
    float* dW,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    if (X == nullptr || dY == nullptr || dW == nullptr || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    // Header-only fallback path: avoids requiring the cuBLAS development header/library at extension build time.
    // The Python training loop uses torch.matmul for dW in default hybrid mode.
    dim3 block(EL_DW_BLOCK_N, EL_DW_BLOCK_K, 1);
    dim3 grid(el_ceil_div_host(N, EL_DW_BLOCK_N), el_ceil_div_host(K, EL_DW_BLOCK_K), 1);

    el_ste_weight_grad_kernel<EL_DW_BLOCK_K, EL_DW_BLOCK_N, EL_DW_STEP_M><<<grid, block, 0, stream>>>(
        X, dY, dW, M, N, K);
    return cudaGetLastError();
}

extern "C" cudaError_t el_bitlinear_backward_async(
    const __half* X,
    const __half* dY,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* dX,
    float* dW,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    cudaError_t err = el_bitlinear_backward_input_async(dY, W_packed, W_scale, dX, M, N, K, stream);
    if (err != cudaSuccess) return err;
    return el_bitlinear_ste_weight_grad_async(X, dY, dW, M, N, K, stream);
}

extern "C" cudaError_t el_bitlinear_backward_input_from_shadow_async(
    const __half* dY,
    const float* W_shadow,
    __half* dX,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    if (dY == nullptr || W_shadow == nullptr || dX == nullptr || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    el_packed_t* W_packed = nullptr;
    float* W_scale = nullptr;
    const size_t wpack_bytes = el_bitlinear_packed_weight_bytes(K, N);
    const size_t wscale_bytes = (size_t)K * sizeof(float);

    cudaError_t alloc0 = cudaMallocAsync((void**)&W_packed, wpack_bytes, stream);
    if (alloc0 != cudaSuccess) return alloc0;
    cudaError_t alloc1 = cudaMallocAsync((void**)&W_scale, wscale_bytes, stream);
    if (alloc1 != cudaSuccess) {
        cudaFreeAsync(W_packed, stream);
        return alloc1;
    }

    cudaError_t err = el_pack_ternary_weights_async(W_shadow, reinterpret_cast<uint32_t*>(W_packed), W_scale, K, N, stream);
    if (err == cudaSuccess) {
        err = el_bitlinear_backward_input_async(dY, reinterpret_cast<const uint32_t*>(W_packed), W_scale, dX, M, N, K, stream);
    }

    cudaError_t free0 = cudaFreeAsync(W_packed, stream);
    cudaError_t free1 = cudaFreeAsync(W_scale, stream);
    if (err != cudaSuccess) return err;
    if (free0 != cudaSuccess) return free0;
    if (free1 != cudaSuccess) return free1;
    return cudaSuccess;
}

extern "C" cudaError_t el_bitlinear_backward_from_shadow_async(
    const __half* X,
    const __half* dY,
    const float* W_shadow,
    __half* dX,
    float* dW,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    if (X == nullptr || dY == nullptr || W_shadow == nullptr || dX == nullptr || dW == nullptr ||
        M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    const int words = el_bitlinear_packed_words_per_row(N);
    el_packed_t* W_packed = nullptr;
    float* W_scale = nullptr;
    const size_t wp_bytes = (size_t)K * (size_t)words * sizeof(el_packed_t);
    const size_t ws_bytes = (size_t)K * sizeof(float);

    cudaError_t alloc0 = cudaMallocAsync((void**)&W_packed, wp_bytes, stream);
    if (alloc0 != cudaSuccess) return alloc0;
    cudaError_t alloc1 = cudaMallocAsync((void**)&W_scale, ws_bytes, stream);
    if (alloc1 != cudaSuccess) {
        cudaFreeAsync(W_packed, stream);
        return alloc1;
    }

    cudaError_t err = el_pack_ternary_weights_async(
        W_shadow, reinterpret_cast<uint32_t*>(W_packed), W_scale, K, N, stream);
    if (err == cudaSuccess) {
        err = el_bitlinear_backward_async(
            X, dY, reinterpret_cast<const uint32_t*>(W_packed), W_scale, dX, dW, M, N, K, stream);
    }

    cudaError_t free0 = cudaFreeAsync(W_packed, stream);
    cudaError_t free1 = cudaFreeAsync(W_scale, stream);
    if (err != cudaSuccess) return err;
    if (free0 != cudaSuccess) return free0;
    if (free1 != cudaSuccess) return free1;
    return cudaSuccess;
}
