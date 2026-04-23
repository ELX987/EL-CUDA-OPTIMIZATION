#include <cuda_fp16.h>
#include <torch/extension.h>

// This is the kernel that runs on the GPU
__global__ void fused_aq_gemv_kernel(
    const int* __restrict__ indices,      // Global VRAM
    const half* __restrict__ codebooks,   // Global VRAM
    const half* __restrict__ inputs,      // Global VRAM
    half* __restrict__ outputs,           // Global VRAM
    int num_codebooks,                    // e.g., 16
    int dict_size,                        // e.g., 256
    int hidden_dim,                       
    int total_codebook_elements           // Added this: total items to load (16 * 256 = 4096)
) {
    // =========================================================================
    // OPTIMIZATION PHASE 1: ALLOCATE SHARED MEMORY
    // Placement: Must be the very first thing inside the kernel.
    // 'extern' tells the GPU that Python will dictate the size during launch.
    // =========================================================================
    extern __shared__ half shared_codebooks[]; 
    
    // Get this specific thread's ID within its team (Block)
    int tid = threadIdx.x; 
    
    // =========================================================================
    // OPTIMIZATION PHASE 2: COLLABORATIVE LOADING
    // Placement: Before the math. 
    // Instead of one thread doing all the work, we split the 4096 codebook 
    // values among the 256 threads in this block. Each thread loads a few 
    // values from slow VRAM into fast Shared Memory simultaneously.
    // =========================================================================
    for (int i = tid; i < total_codebook_elements; i += blockDim.x) {
        shared_codebooks[i] = codebooks[i];
    }
    
    // =========================================================================
    // OPTIMIZATION PHASE 3: THE BARRIER
    // Placement: Immediately after the loading loop.
    // CRITICAL: We must freeze all 256 threads right here. No thread is allowed 
    // to move forward until the entire codebook is fully loaded into the cache.
    // =========================================================================
    __syncthreads();

    // =========================================================================
    // PHASE 4: THE MATH (Using the fast memory)
    // =========================================================================
    // Now we calculate which row of the matrix this thread is responsible for
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < hidden_dim) {
        // This accumulator goes into an ultra-fast hardware Register
        float sum = 0.0f; 
        
        for (int col = 0; col < hidden_dim; ++col) {
            float reconstructed_weight = 0.0f;
            
            for (int m = 0; m < num_codebooks; ++m) {
                // Get the compressed index
                int dict_idx = indices[row * hidden_dim * num_codebooks + col * num_codebooks + m];
                
                // PULL FROM SHARED MEMORY!
                // We use shared_codebooks instead of the global codebooks array.
                // This is what keeps your Blackwell Tensor Cores fed at maximum speed.
                reconstructed_weight += __half2float(shared_codebooks[m * dict_size + dict_idx]); 
            }
            
            float input_val = __half2float(inputs[col]);
            sum += reconstructed_weight * input_val;
        }
        
        // Write the final computed value back to slow VRAM
        outputs[row] = __float2half(sum); 
    }
}

// ... (The PyTorch C++ wrapper below remains the same as before) ...
// This is the C++ wrapper that PyTorch will call
torch::Tensor aq_gemv_forward(torch::Tensor indices, torch::Tensor codebooks, torch::Tensor inputs) {
    // ... (Setup output tensor sizes and launch the __global__ kernel above) ...
}

// Bind it to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &aq_gemv_forward, "Fused AQ Forward Pass");
}