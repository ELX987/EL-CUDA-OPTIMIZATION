#include <torch/extension.h>
#include <cstdint>
#include <vector>

// Forward declare the launcher with exactly 14 arguments
void launch_aq_packed_kernel(
    const uint8_t* indices,
    const at::Half* codebooks,
    const at::Half* inputs,
    at::Half* outputs,
    int blocks_x,             // 5
    int threads,              // 6
    int shared_mem_size,      // 7
    int num_codebooks,        // 8
    int dict_size,            // 9
    int in_features,          // 10
    int out_features,         // 11
    int stride,               // 12
    int total_codebook_elements, // 13
    int num_tokens            // 14
);

torch::Tensor aq_gemv_forward(torch::Tensor packed_indices, torch::Tensor codebooks, torch::Tensor inputs) {
    int in_features = inputs.size(-1); 
    auto inputs_2d = inputs.view({-1, in_features}).contiguous();
    
    int num_tokens = inputs_2d.size(0);
    int out_features = packed_indices.size(0);
    int num_codebooks = codebooks.size(0);
    int dict_size = codebooks.size(1);
    int total_codebook_elements = num_codebooks * dict_size;
    int stride = (packed_indices.size(1) / num_codebooks); 

    auto packed_indices_c = packed_indices.contiguous();
    auto codebooks_c = codebooks.contiguous();

    std::vector<int64_t> output_shape = inputs.sizes().vec();
    output_shape.back() = out_features; 
    auto outputs = torch::empty(output_shape, inputs.options().dtype(torch::kHalf));
    auto outputs_2d = outputs.view({-1, out_features});

    int threads = 256;
    // This is now blocks_y because it maps to the feature dimension
    int blocks_y = (out_features + threads - 1) / threads;
    size_t shared_mem_size = (total_codebook_elements * sizeof(at::Half)) + (243 * 5 * sizeof(int8_t));

    launch_aq_packed_kernel(
        packed_indices_c.data_ptr<uint8_t>(),
        codebooks_c.data_ptr<at::Half>(),
        inputs_2d.data_ptr<at::Half>(),
        outputs_2d.data_ptr<at::Half>(),
        blocks_y, threads, (int)shared_mem_size,
        num_codebooks, dict_size, in_features, out_features, stride, 
        total_codebook_elements, num_tokens
    );

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &aq_gemv_forward, "AQLoRA 2D Blackwell Kernel");
}
