import os
import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

# --- 1. CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
SAVE_DIR = "/workspace/Qwen3.6-35B-AQ-PACKED"
os.environ["HF_TOKEN"] = "HF_TOKEN"

# Create clean directory
if os.path.exists(SAVE_DIR):
    import shutil
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. THE TRUE BASE-3 FRACTIONAL PACKER ---
def pack_ternary_weights(weight):
    """
    Quantizes FP16 weights to 1.58-bit Ternary (-1, 0, 1) and packs 
    every 5 values into a single uint8 byte using Base-3 math.
    """
    with torch.no_grad():
        out_features, in_features = weight.shape
        
        # 1. Calculate precise layer volume (Mean of Absolute Values)
        # This becomes the 'scales' tensor that restores the neural magnitude
        scales = weight.abs().mean(dim=1).clamp(min=1e-6)
        
        # 2. Normalize and apply 1.58-bit Ternary Quantization
        w_norm = (weight / scales.unsqueeze(1)).round().clamp(-1, 1).to(torch.int8)
        
        # 3. Geometry padding (Kernel expects multiples of 5)
        pad_len = (5 - (in_features % 5)) % 5
        if pad_len > 0:
            w_norm = torch.nn.functional.pad(w_norm, (0, pad_len))
        
        num_bytes = w_norm.shape[1] // 5
        
        # 4. Map ternary (-1, 0, 1) to Base-3 integers (0, 1, 2)
        w_val = w_norm + 1 
        w_groups = w_val.view(out_features, num_bytes, 5)
        
        # 5. Base-3 Compression
        coeffs = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16, device=weight.device)
        packed_indices = (w_groups * coeffs).sum(dim=2).to(torch.uint8)
        
        # 6. Generate the pure Dictionary Codebook
        # Since we use scales to dictate volume, the codebook is strictly -1.0 and 1.0
        codebooks = torch.tensor([[-1.0, 1.0]], dtype=torch.float16, device=weight.device)
        
        # Return CPU tensors to prevent VRAM accumulation
        return packed_indices.cpu(), codebooks.cpu(), scales.float().cpu()

# --- 3. LOAD MODEL SKELETON ---
print(f"💀 Loading {MODEL_ID} into CPU Memory...")
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    config=config,
    torch_dtype=torch.float16, 
    device_map="cpu", # Keep 35B on CPU, only move layers to GPU for packing
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# --- 4. PRODUCTION PACKING LOOP ---
print("\nStarting True Base-3 Quantization (40 Layers)...")
for i, layer in tqdm(enumerate(model.model.layers), total=model.config.num_hidden_layers):
    layer_dir = os.path.join(SAVE_DIR, f"layer_{i}")
    os.makedirs(layer_dir, exist_ok=True)
    
    # A. PROCESS SHARED EXPERT
    shared_projs = {
        "gate": layer.mlp.shared_expert.gate_proj,
        "up": layer.mlp.shared_expert.up_proj,
        "down": layer.mlp.shared_expert.down_proj
    }
    
    for name, proj in shared_projs.items():
        w_cuda = proj.weight.data.to("cuda")
        idx, cb, s = pack_ternary_weights(w_cuda)
        torch.save({"idx": idx, "cb": cb, "s": s}, f"{layer_dir}/shared_{name}.pt")
        del w_cuda

    # B. PROCESS STACKED ROUTED EXPERTS
    expert_block = layer.mlp.experts
    for j in range(expert_block.num_experts):
        
        # The base model fuses Gate and Up. We must slice them in half.
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
        
        del fused, w_gate, w_up, w_down

    # Memory Management
    layer.mlp = None
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n✅ Packing Complete! 1.6-bit Base-3 weights saved to {SAVE_DIR}")
