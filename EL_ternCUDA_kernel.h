#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t el_pack_ternary_weights_async(
    const float* W_shadow,
    uint32_t* W_packed,
    float* W_scale,
    int K,
    int N,
    cudaStream_t stream);

cudaError_t el_dequantize_packed_to_fp16_async(
    const uint32_t* W_packed,
    const float* W_scale,
    __half* W_deq,
    int K,
    int N,
    cudaStream_t stream);

cudaError_t el_quantize_fp16_per_row_int8_async(
    const __half* X,
    int8_t* X_q,
    float* X_scale,
    int rows,
    int cols,
    cudaStream_t stream);

cudaError_t el_bitlinear_forward_prequantized_async(
    const int8_t* X_q,
    const float* X_scale,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* Y,
    int32_t* Acc_optional,
    int M,
    int N,
    int K,
    cudaStream_t stream);

cudaError_t el_bitlinear_forward_async(
    const __half* X,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* Y,
    int M,
    int N,
    int K,
    cudaStream_t stream);

cudaError_t el_bitlinear_forward_from_shadow_async(
    const __half* X,
    const float* W_shadow,
    __half* Y,
    int M,
    int N,
    int K,
    cudaStream_t stream);



cudaError_t el_bitlinear_backward_input_async(
    const __half* dY,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* dX,
    int M,
    int N,
    int K,
    cudaStream_t stream);

cudaError_t el_bitlinear_backward_input_quantized_async(
    const __half* dY,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* dX,
    int M,
    int N,
    int K,
    cudaStream_t stream);

cudaError_t el_bitlinear_ste_weight_grad_async(
    const __half* X,
    const __half* dY,
    float* dW,
    int M,
    int N,
    int K,
    cudaStream_t stream);

cudaError_t el_bitlinear_backward_async(
    const __half* X,
    const __half* dY,
    const uint32_t* W_packed,
    const float* W_scale,
    __half* dX,
    float* dW,
    int M,
    int N,
    int K,
    cudaStream_t stream);


cudaError_t el_bitlinear_backward_input_from_shadow_async(
    const __half* dY,
    const float* W_shadow,
    __half* dX,
    int M,
    int N,
    int K,
    cudaStream_t stream);

cudaError_t el_bitlinear_backward_from_shadow_async(
    const __half* X,
    const __half* dY,
    const float* W_shadow,
    __half* dX,
    float* dW,
    int M,
    int N,
    int K,
    cudaStream_t stream);

int el_bitlinear_packed_words_per_row(int N);
size_t el_bitlinear_packed_weight_bytes(int K, int N);
size_t el_bitlinear_forward_temp_bytes(int M, int N);
size_t el_bitlinear_forward_from_shadow_temp_bytes(int M, int N, int K);
size_t el_bitlinear_backward_from_shadow_temp_bytes(int N, int K);

#ifdef __cplusplus
}
#endif
