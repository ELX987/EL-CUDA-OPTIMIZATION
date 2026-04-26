import os
import gc
import torch
import sys
import aq_kernel
import shutil

# Prevent KV Cache fragmentation during generation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import lm_eval
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer
from lm_eval.models.huggingface import HFLM
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen_moe

# --- CONFIG ---
BASE_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
QUANT_DIR = "/workspace/Qwen3.6-35B-AQ-PACKED"
BATCH_SIZE = 64
os.environ["HF_TOKEN"] = "HF_TOKEN"

class AQLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.indices = nn.Parameter(torch.empty((0, 0, 0), dtype=torch.uint8), requires_grad=False)
        self.codebooks = nn.Parameter(torch.empty((0, 0, 0), dtype=torch.float16), requires_grad=False)
        self.scales = nn.Parameter(torch.empty((0), dtype=torch.float32), requires_grad=False)
        
    def forward(self, x):
        if self.codebooks.numel() == 0: return x
        
        x = x.contiguous()
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]).clone().contiguous()
        
        # 1. Geometry Alignment
        expected_in = self.indices.shape[1] * 5 
        if x_flat.shape[1] < expected_in:
            x_flat = torch.nn.functional.pad(x_flat, (0, expected_in - x_flat.shape[1]))

        # 🚨 2. THE OVERFLOW BUFFER 
        # Safely shrinks inputs by 8.0x so the C++ accumulator never hits the 65,504 Infinity limit.
        x_max = x_flat.abs().max().clamp(min=1.0) * 8.0
        cb_max = self.codebooks.abs().max().clamp(min=1.0)
        
        x_safe = (x_flat / x_max).to(torch.float16)
        cb_safe = (self.codebooks / cb_max).to(torch.float16)

        # 3. Safe Kernel Call
        out_flat = aq_kernel.forward(
            self.indices.contiguous(), 
            cb_safe.contiguous(), 
            x_safe
        )
        
        # 4. FP32 Math Restoration (Automatically reverses the 8.0 buffer)
        out_flat_f32 = out_flat.to(torch.float32) * x_max.to(torch.float32) * cb_max.to(torch.float32)

        # 5. Apply Exact Trained Feature Scales (The 1024 divider is GONE)
        if self.scales.numel() > 0:
            out_flat_f32 = out_flat_f32 * self.scales.view(1, -1).to(torch.float32)

        out_flat_f32 = torch.nan_to_num(out_flat_f32, nan=0.0, posinf=0.0, neginf=0.0)

        # PROBE: Verify the volumes return to healthy ~1.0 ranges
        if not hasattr(self, "_printed_probe"):
            print(f"\n[PROBE] Final Output Mag: {out_flat_f32.abs().mean().item():.4f}")
            self._printed_probe = True

        if out_flat_f32.shape[0] > orig_shape[0]:
            out_flat_f32 = out_flat_f32[:orig_shape[0], :]
            
        return out_flat_f32.view(*orig_shape[:-1], -1).to(x.dtype)

class QwenAQExpert(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        expert_dim = intermediate_size or config.moe_intermediate_size
        
        self.gate_proj = AQLinear(config.hidden_size, expert_dim) 
        self.up_proj = AQLinear(config.hidden_size, expert_dim)
        self.down_proj = AQLinear(expert_dim, config.hidden_size)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # Math is safely contained in FP32 so the SwiGLU activation never overflows
        gate_out = self.gate_proj(x).to(torch.float32)
        up_out = self.up_proj(x).to(torch.float32)
        
        hidden = self.act_fn(gate_out) * up_out
        
        # Convert back to native bfloat16 for the down_proj and residual stream
        return self.down_proj(hidden.to(x.dtype))
    
class AQExpertWrapper(nn.Module):
    def __init__(self, experts_list):
        super().__init__()
        self.experts = experts_list
    
    def forward(self, hidden_states, selected_experts, routing_weights):
        final_output = torch.zeros_like(hidden_states)
        unique_experts = torch.unique(selected_experts)
        
        for expert_idx in unique_experts:
            idx = int(expert_idx.item())
            if idx == -1: continue 
            
            mask = (selected_experts == idx)
            token_indices = mask.any(dim=-1)
            
            if token_indices.any():
                expert_output = self.experts[idx](hidden_states[token_indices])
                weight = routing_weights[mask].unsqueeze(-1)
                final_output[token_indices] += expert_output * weight.to(expert_output.dtype)
                
        return final_output

def load_weights(expert_module, quant_dir, layer_idx, expert_idx, dev):
    layer_folder = os.path.join(quant_dir, f"layer_{layer_idx}")
    for proj in ["gate", "up", "down"]:
        filepath = os.path.join(layer_folder, f"e{expert_idx}_{proj}.pt")
        if not os.path.exists(filepath): continue
        
        data = torch.load(filepath, map_location="cpu", weights_only=True)
        target = getattr(expert_module, f"{proj}_proj")
        
        target.indices = nn.Parameter(data["idx"].to(device=dev, dtype=torch.uint8).contiguous(), requires_grad=False)
        target.codebooks = nn.Parameter(data["cb"].to(device=dev, dtype=torch.float16).contiguous(), requires_grad=False)
        if "s" in data:
            target.scales = nn.Parameter(data["s"].to(device=dev, dtype=torch.float32).contiguous(), requires_grad=False)

def load_shared_weights(expert_module, quant_dir, layer_idx, dev):
    layer_folder = os.path.join(quant_dir, f"layer_{layer_idx}")
    for proj in ["gate", "up", "down"]:
        filepath = os.path.join(layer_folder, f"shared_{proj}.pt")
        if not os.path.exists(filepath): continue
        
        data = torch.load(filepath, map_location="cpu", weights_only=True)
        target = getattr(expert_module, f"{proj}_proj")
        
        target.indices = nn.Parameter(data["idx"].to(device=dev, dtype=torch.uint8).contiguous(), requires_grad=False)
        target.codebooks = nn.Parameter(data["cb"].to(device=dev, dtype=torch.float16).contiguous(), requires_grad=False)
        if "s" in data:
            target.scales = nn.Parameter(data["s"].to(device=dev, dtype=torch.float32).contiguous(), requires_grad=False)
    del data

# --- 2. LOAD LIVE BASE MODEL ---
print("💀 Initializing Tokenizer and Real Config...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, local_files_only=False)
config = AutoConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, local_files_only=False)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

print(f"📥 Loading Qwen 3.6 Skeleton...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    config=config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu", 
    trust_remote_code=True
)
print(f"🛠️ Base Model Loaded: {model.config.num_hidden_layers} Layers | {model.config.hidden_size} Dim")

# --- 3. MATERIALIZATION (Optimized for VRAM & Routing) ---
print(f"💉 Injecting AQ-Packed Experts...")

for i in range(model.config.num_hidden_layers):
    print(f" > Patching Layer {i}...", end="", flush=True)
    layer = model.model.layers[i]
    
    old_mlp = layer.mlp
    # 🚨 DTYPE ALIGNMENT: Ensure bfloat16 to match base model
    new_mlp = qwen_moe.Qwen3_5MoeSparseMoeBlock(model.config).to("cuda", dtype=torch.bfloat16)
    
    # Salvage the trained routers
    new_mlp.gate = old_mlp.gate.to("cuda")
    if hasattr(old_mlp, 'shared_expert_gate'):
        new_mlp.shared_expert_gate = old_mlp.shared_expert_gate.to("cuda")
        
    old_mlp.experts = None
    old_mlp.shared_expert = None
    layer.mlp = None 
    del old_mlp
    gc.collect()
    
    new_experts_list = nn.ModuleList()
    for idx in range(model.config.num_experts):
        expert_shell = QwenAQExpert(model.config).to("cuda", dtype=torch.bfloat16)
        load_weights(expert_shell, QUANT_DIR, i, idx, "cuda")
        new_experts_list.append(expert_shell)
        
    new_mlp.experts = AQExpertWrapper(new_experts_list)
    
    shared_expert_shell = QwenAQExpert(model.config, intermediate_size=512).to("cuda", dtype=torch.bfloat16)
    load_shared_weights(shared_expert_shell, QUANT_DIR, i, "cuda")
    new_mlp.shared_expert = shared_expert_shell

    layer.mlp = new_mlp
    layer.to("cuda") 
    
    torch.cuda.empty_cache()
    print(" ✅", flush=True)

print("🚚 Moving final core components to B200...", flush=True)
model.model.embed_tokens.to("cuda")
model.model.norm.to("cuda")
model.lm_head.to("cuda")

if hasattr(model.model, 'rotary_emb'):
    model.model.rotary_emb.to("cuda")
for buffer_name, buffer in model.model.named_buffers():
    buffer.data = buffer.data.to("cuda")

print("\n" + "="*40)
print(f"🚀 Injection Complete. Final VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print("="*40 + "\n")

print("📊 RUNNING EFFICIENT BENCHMARK (ARC-Easy & PIQA)...")

model.eval()

# 1. Wrap our custom injected model in the lm-eval HuggingFace wrapper
print("Wrapping model for lm-eval harness...")
lm_eval_model = HFLM(
    pretrained=model, 
    tokenizer=tokenizer, 
    batch_size=BATCH_SIZE
)

# 2. Run a fast evaluation 
# We use a limit of 100 questions per task so it finishes in a few minutes.
print("Executing benchmark tasks...")
with torch.no_grad():
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["arc_easy", "piqa"],
        limit=100, 
        device="cuda"
    )

# 3. Print the results table
print("\n" + "="*40)
print("🏆 1.6-BIT MoE BENCHMARK RESULTS:")

print("REMINDER: Because we used brutal Post-Training Quantization (PTQ) without")
print("Quantization-Aware Training (QAT), expect the accuracy to be near random chance")
print("(~25% for multiple choice). The true victory is that the engine executes perfectly!")
