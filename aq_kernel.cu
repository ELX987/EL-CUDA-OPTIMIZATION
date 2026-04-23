#include <cuda_bf16.h>
#include <torch/extension.h>

// =========================================================================
// 1. THE GPU KERNEL (BFloat16 & Blackwell Optimized)
// =========================================================================
__global__ void fused_aq_gemm_kernel(
    const int* __restrict__ indices,
    const __nv_bfloat16* __restrict__ codebooks,
    const __nv_bfloat16* __restrict__ inputs,
    __nv_bfloat16* __restrict__ outputs,
    int num_tokens,
    int in_features,
    int out_features,
    int num_codebooks,
    int dict_size,
    int vector_dim
) {
    extern __shared__ __nv_bfloat16 shared_codebooks[]; 
    
    int tid = threadIdx.x; 
    int total_cb_elements = num_codebooks * dict_size * vector_dim;

    // Load codebooks into ultra-fast L1 Shared Memory
    for (int i = tid; i < total_cb_elements; i += blockDim.x) {
        shared_codebooks[i] = codebooks[i];
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = in_features / vector_dim;

    if (row < out_features) {
        for (int t = 0; t < num_tokens; ++t) {
            float sum = 0.0f; 
            
            for (int block = 0; block < num_blocks; ++block) {
                for (int v = 0; v < vector_dim; ++v) {
                    int col = block * vector_dim + v;
                    float reconstructed_weight = 0.0f;
                    
                    for (int m = 0; m < num_codebooks; ++m) {
                        int idx_offset = row * (num_blocks * num_codebooks) + block * num_codebooks + m;
                        int dict_idx = indices[idx_offset];
                        int cb_offset = m * (dict_size * vector_dim) + dict_idx * vector_dim + v;
                        
                        reconstructed_weight += __bfloat162float(shared_codebooks[cb_offset]); 
                    }
                    
                    float input_val = __bfloat162float(inputs[t * in_features + col]);
                    sum += reconstructed_weight * input_val;
                }
            }
            outputs[t * out_features + row] = __float2bfloat16(sum); 
        }
    }
}

// =========================================================================
// 2. THE PYTORCH C++ WRAPPER
// =========================================================================
torch::Tensor aq_gemm_forward(torch::Tensor indices, torch::Tensor codebooks, torch::Tensor inputs) {
    int num_tokens = inputs.size(0);
    int in_features = inputs.size(1);
    int out_features = indices.size(0);
    
    int num_codebooks = codebooks.size(0);
    int dict_size = codebooks.size(1);
    int vector_dim = codebooks.size(2);

    auto outputs = torch::empty({num_tokens, out_features}, inputs.options());

    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;
    
    // Define the memory size needed for the hardware override
    int shared_mem_bytes = num_codebooks * dict_size * vector_dim * sizeof(uint16_t);

    // UNLOCK BLACKWELL SHARED MEMORY LIMIT
    cudaFuncSetAttribute(
        (const void*)fused_aq_gemm_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shared_mem_bytes
    );

    // Launch the kernel
    fused_aq_gemm_kernel<<<blocks, threads, shared_mem_bytes>>>(
        indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(codebooks.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(inputs.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(outputs.data_ptr<at::BFloat16>()),
        num_tokens, in_features, out_features, num_codebooks, dict_size, vector_dim
    );

    return outputs;
}

// =========================================================================
// 3. PYTHON BINDING
// =========================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &aq_gemm_forward, "Fused AQ Forward Pass (BFloat16)");
}
