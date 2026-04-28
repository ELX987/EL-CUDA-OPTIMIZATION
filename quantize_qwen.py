import os
import gc
import torch
import torch.nn as nn 
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use the number of available GPUs or a defined limit
MAX_WORKERS = 8  


# --- 1. CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
SAVE_DIR = "/workspace/Qwen3.6-35B-AQ-PACKED"
os.environ["HF_TOKEN"] = "HF_TOKEN"


# Create clean directory
if os.path.exists(SAVE_DIR):
    import shutil
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)


# --- 2. CORE QUANTIZATION WORKER FUNCTION (Isolated for Parallel Execution) ---
def pack_ternary_weights(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes FP16 weights to 1.58-bit Ternary (-1, 0, 1) and packs 
    every 5 values into a single uint8 byte using Base-3 math.
    This function is designed to be self-contained for parallel workers.
    """
    # Ensure the input weight tensor is handled on CPU/GPU as needed by the worker context.
    weight = weight.detach().cpu() 


    with torch.no_grad():
        out_features, in_features = weight.shape
        
        scales = weight.abs().mean(dim=1).clamp(min=1e-6) # Scales should be CPU float for saving
        w_norm = (weight / scales.unsqueeze(1)).round().clamp(-1, 1).to(torch.int8)
        
        pad_len = (5 - (in_features % 5)) % 5
        if pad_len > 0:
            w_norm = torch.nn.functional.pad(w_norm, (0, pad_len))
        
        num_bytes = w_norm.shape[1] // 5
        
        # Map ternary (-1, 0, 1) to Base-3 integers (0, 1, 2)
        w_val = w_norm + 1 
        w_groups = w_val.view(out_features, num_bytes, 5)
        coeffs = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16)
        packed_indices = (w_groups * coeffs).sum(dim=2).to(torch.uint8)
        
        codebooks = torch.tensor([[-1.0, 1.0]], dtype=torch.float16) # Always CPU float for saving
        
        # Return CPU tensors to prevent VRAM accumulation and ensure portability across processes
        return packed_indices.cpu(), codebooks.cpu(), scales.float().cpu()


def process_layer(args: dict):
    """Worker function that performs quantization for a single layer."""
    i = args['index']
    layer = args['layer']
    layer_dir = os.path.join(SAVE_DIR, f"layer_{i}")
    os.makedirs(layer_dir, exist_ok=True)


    # Using a print statement here is fine, but note that in parallel workers, 
    # interleaved output can make logging messy. Use the tqdm progress bar for tracking.
    print(f"\n[Worker {i}] Starting processing for Layer {i}...")


    # A. PROCESS SHARED EXPERT
    shared_projs = {
        "gate": layer.mlp.shared_expert.gate_proj,
        "up": layer.mlp.shared_expert.up_proj,
        "down": layer.mlp.shared_expert.down_proj
    }
    
    for name, proj in shared_projs.items():
        # Move to CUDA only for the computation required by the worker
        w_cuda = proj.weight.data.to("cuda") 
        idx, cb, s = pack_ternary_weights(w_cuda)
        torch.save({"idx": idx, "cb": cb, "s": s}, f"{layer_dir}/shared_{name}.pt")


    # B. PROCESS STACKED ROUTED EXPERTS
    expert_block = layer.mlp.experts
    for j in range(expert_block.num_experts):
        
        fused = expert_block.gate_up_proj.data[j].to("cuda")
        mid = fused.shape[0] // 2
        
        w_gate = fused[:mid, :]
        w_up = fused[mid:, :]
        w_down = expert_block.down_proj.data[j].to("cuda")
        
        # Pack and save Gate
        idx, cb, s = pack_ternary_weights(w_gate)
        torch.save({"idx": idx, "cb": cb, "s": s}, f"{layer_dir}/e{j}_gate.pt")
        
        # Pack and save Up
        idx, cb, s = pack_ternary_weights(w_up)
        torch.save({"idx": idx, "cb": cb, "s": s}, f"{layer_dir}/e{j}_up.pt")
        
        # Pack and save Down
        idx, cb, s = pack_ternary_weights(w_down)
        torch.save({"idx": idx, "cb": cb, "s": s}, f"{layer_dir}/e{j}_down.pt")


    print(f"[Worker {i}] Successfully finished quantization.")
    return True # Return status for tracking


# --- 3. LOAD MODEL SKELETON (Main Execution Logic) ---
def main():
    """The main orchestration function to load the model and distribute quantization work."""
    print("==============================================")
    print(f"🚀 STARTING QUANTIZATION OF {MODEL_ID}...")
    print("==============================================")
    
    # 1. Load Model Skeleton on CPU (Memory saving technique)
    print(f"\n💀 Loading {MODEL_ID} into CPU Memory...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        config=config,
        torch_dtype=torch.float16, 
        device_map="cpu", # Crucial: Keep the model on CPU during loading to save VRAM
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # --- Prepare Tasks for Parallel Workers ---
    print("\nPreparing quantization tasks...")
    all_layers = list(model.model.layers)
    total_layers = len(all_layers)


    # Create a list of dictionaries, where each dict holds the necessary context for one worker
    tasks = []
    for i, layer in enumerate(all_layers):
        tasks.append({'index': i, 'layer': layer})

    print(f"Total layers found: {total_layers}. Starting parallel quantization...")


    # --- PARALLEL EXECUTION USING THREAD POOL ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_layer, args) for args in tasks]
        results = []

        print("\n--- Monitoring Progress (This may take several hours...) ---\n")
        # The tqdm iterator uses the future object to track completion status
        for future in tqdm(as_completed(futures), total=total_layers, desc="Quantizing Layers"):
            try:
                result = future.result()
                if result is True:
                    results.append("Success")
            except Exception as e:
                print(f"\n⚠️ CRITICAL FAILURE in a worker process: {e}")


    # --- 4. FINAL CLEANUP AND SAVE ---

    del model # Free up all loaded weights from Python memory scope on the CPU RAM
    gc.collect()
    torch.cuda.empty_cache()

    print("\n==============================================")
    if len(results) == total_layers:
        print("✅ QUANTIZATION SUCCESS!")
        print(f"All {total_layers} layers were processed successfully.")
        print(f"Quantized weights (1.58-bit Base-3) are saved to: {SAVE_DIR}")
    else:
        print("❌ WARNING: Quantization failed for some layers. Check logs above.")

if __name__ == "__main__":
    # Ensure CUDA is available before running the main quantization loop
    if not torch.cuda.is_available():
        print("\n=====================================================")
        print("FATAL ERROR: CUDA device not found.")
        print("This script requires a GPU for weight transfer and calculation.")
        print("Please ensure your PyTorch environment is configured correctly.")
        print("=====================================================")
    else:
        main()
