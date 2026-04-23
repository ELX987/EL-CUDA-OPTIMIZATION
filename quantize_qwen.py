import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- 1. CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
BASE_SAVE_DIR = "./quant_weights"
NUM_CODEBOOKS = 16
DICT_SIZE = 256
VECTOR_DIM = 8

# --- 2. NATIVE BLACKWELL K-MEANS ---
def quantize_with_hessian(weight, hessian_diag, num_codebooks=16, dict_size=256, vector_dim=8, niter=20):
    if hessian_diag is None:
        hessian_diag = torch.ones(weight.shape[1], device=weight.device)
    
    # Scale by Hessian importance
    scales = torch.sqrt(hessian_diag + 1e-6)
    scaled_weight = weight * scales.unsqueeze(0)
    
    orig_shape = weight.shape
    x = scaled_weight.reshape(-1, vector_dim).float()
    n_vectors = x.shape[0]
    
    indices_list, codebooks_list = [], []
    residual = x.clone()
    
    for m in range(num_codebooks):
        # Initialize centroids randomly from residual
        rand_idx = torch.randperm(n_vectors, device=x.device)[:dict_size]
        centroids = residual[rand_idx].clone()
        
        for _ in range(niter):
            # Fast Euclidean distance via cuBLAS
            dists = torch.cdist(residual, centroids)
            labels = dists.argmin(dim=-1)
            
            # Update Centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(dict_size, 1, device=x.device)
            new_centroids.index_add_(0, labels, residual)
            counts.index_add_(0, labels, torch.ones(n_vectors, 1, device=x.device))
            centroids = new_centroids / (counts + 1e-6)
            
        codebooks_list.append(centroids.to(torch.float16))
        indices_list.append(labels.to(torch.int32))
        residual -= centroids[labels]

    return torch.stack(indices_list, dim=-1).view(orig_shape[0], -1, num_codebooks), \
           torch.stack(codebooks_list, dim=0), \
           scales

# --- 3. LOAD & CALIBRATE ---
print(f"Initializing {MODEL_ID} on Blackwell GPUs...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

hessian_diagonals = {}
def get_hook(name):
    def hook(m, i, o):
        act = i[0].detach().float()
        sq = (act ** 2).sum(dim=(0, 1))
        hessian_diagonals[name] = hessian_diagonals.get(name, 0) + sq
    return hook

print("Capturing Hessian importance via Python coding samples...")
hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.mlp.shared_expert.register_forward_hook(get_hook(f"l{i}_shared")))
    hooks.append(layer.mlp.experts.register_forward_hook(get_hook(f"l{i}_stacked")))

# High-quality Python calibration data
calib_data = [
    "import torch\nimport numpy as np\ndef rsi(prices, n=14):",
    "class BlackwellAQ(nn.Module):\n    def __init__(self, config):",
    "def train_moe(model, loader):\n    for batch in loader:\n        logits = model(batch)"
]
for text in calib_data:
    model(**tokenizer(text, return_tensors="pt").to("cuda"))
for h in hooks: h.remove()

# --- 4. PRODUCTION LOOP ---
print("\nStarting Full-Scale Quantization (40 Layers)...")
for i, layer in tqdm(enumerate(model.model.layers), total=40):
    layer_dir = os.path.join(BASE_SAVE_DIR, f"layer_{i}")
    os.makedirs(layer_dir, exist_ok=True)
    
    # A. SHARED EXPERT
    h_shared = hessian_diagonals.get(f"l{i}_shared")
    for name, proj in [("gate", layer.mlp.shared_expert.gate_proj), 
                       ("up", layer.mlp.shared_expert.up_proj), 
                       ("down", layer.mlp.shared_expert.down_proj)]:
        idx, cb, s = quantize_with_hessian(proj.weight.data, h_shared)
        torch.save({"idx": idx, "cb": cb, "s": s}, f"{layer_dir}/shared_{name}.pt")

    # B. STACKED EXPERTS
    expert_block = layer.mlp.experts
    h_stacked = hessian_diagonals.get(f"l{i}_stacked")
    
    for j in range(expert_block.num_experts):
        # Gate & Up
        fused = expert_block.gate_up_proj.data[j]
        mid = fused.shape[0] // 2
        
        # Quantize slices
        for name, weight in [("gate", fused[:mid, :]), ("up", fused[mid:, :]), ("down", expert_block.down_proj.data[j])]:
            idx, cb, s = quantize_with_hessian(weight, h_stacked)
            torch.save({"idx": idx, "cb": cb, "s": s}, f"{layer_dir}/e{j}_{name}.pt")

    torch.cuda.empty_cache()

print(f"\nDone! All weights saved to {BASE_SAVE_DIR}")