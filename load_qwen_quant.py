import os
import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
import torch.utils.checkpoint
from torch.utils.cpp_extension import load

# --- 0. JIT COMPILE THE CUDA KERNEL ---
print("Compiling Blackwell AQ Kernel (this will take ~1-2 minutes on the first run)...")
aq_kernel = load(
    name="aq_kernel",
    sources=["aq_kernel.cu"],  # <--- CHANGE THIS TO THE ACTUAL NAME OF YOUR .cu FILE
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3", 
        "--use_fast_math"
        # PyTorch will auto-detect your dual Pro 6000s and compile for the correct architecture
    ]
)
print("Kernel compilation successful!")

# --- 1. OPTIMIZED ARCHITECTURE ---

class AQExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.indices = nn.Parameter(torch.empty((0, 0, 0), dtype=torch.int32), requires_grad=False)
        self.codebooks = nn.Parameter(torch.empty((0, 0, 0), dtype=torch.bfloat16), requires_grad=False)
        self.scales = nn.Parameter(torch.empty((0), dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        
        x_scaled = x * self.scales.to(dtype=x.dtype).view(1, -1)
        
        # THE FIX: .contiguous() physically aligns the RAM so C++ can read it safely
        x_flat = x_scaled.view(-1, orig_shape[-1]).contiguous()
        
        out_flat = aq_kernel.forward(self.indices, self.codebooks, x_flat)
        
        out = out_flat.view(*orig_shape[:-1], -1)
        
        return torch.clamp(out, min=-50.0, max=50.0)

class LoRAAQExpert(nn.Module):
    def __init__(self, hidden_dim, r=128, lora_alpha=256):
        super().__init__()
        self.gate, self.up, self.down = AQExpert(), AQExpert(), AQExpert()
        
        # FP32 LoRA adapters to catch micro-gradients
        self.lora_A = nn.Linear(hidden_dim, r, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(r, hidden_dim, bias=False, dtype=torch.float32)
        self.scaling = lora_alpha / r
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 1. 1-Bit Path (Scrubbed and Hardcoded to 1% Volume)
        q_gate = torch.nan_to_num(self.gate(x), nan=0.0)
        q_up = torch.nan_to_num(self.up(x), nan=0.0)
        q_mid = torch.nn.functional.silu(q_gate) * q_up
        q_out = self.down(q_mid)

        # Armor-plate the output
        q_out = torch.nan_to_num(q_out, nan=0.0, posinf=10.0, neginf=-10.0)
        q_out = torch.clamp(q_out, min=-50.0, max=50.0)

        # 2. Clean FP32 LoRA Path
        x_fp32 = x.to(torch.float32)
        lora_out = self.lora_B(self.lora_A(x_fp32)) * self.scaling

        # 3. Recombine (No scalar gradients to explode!)
        out = (q_out * 0.01) + lora_out.to(q_out.dtype)
        
        return out.to(x.dtype)

def load_aq_weights_into_expert(expert_module, weights_dir, layer_idx):
    layer_path = os.path.join(weights_dir, f"layer_{layer_idx}")
    target_device = expert_module.lora_A.weight.device
    for name in ["gate", "up", "down"]:
        file_path = os.path.join(layer_path, f"shared_{name}.pt")
        if os.path.exists(file_path):
            data = torch.load(file_path, map_location=target_device)
            target = getattr(expert_module, name)
            target.indices = nn.Parameter(data['idx'].to(target_device), requires_grad=False)
            target.codebooks = nn.Parameter(data['cb'].to(target_device, dtype=torch.bfloat16), requires_grad=False)
            target.scales = nn.Parameter(data['s'].to(target_device, dtype=torch.float32), requires_grad=False)

# --- 2. MODEL INITIALIZATION (Clean & Balanced) ---

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
print(f"Loading {MODEL_ID} shell...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Define balanced device map
device_map = {
    "model.embed_tokens": 0,
    "model.rotary_emb": 0,
    "model.norm": 1,
    "lm_head": 1
}
# Split 40 layers: 0-17 on GPU0, 18-39 on GPU1
for i in range(40):
    device_map[f"model.layers.{i}"] = 0 if i < 18 else 1

# Load the model ONCE with the map
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    dtype=torch.bfloat16, 
    device_map=device_map
)

HIDDEN_SIZE = model.config.hidden_size
NUM_LAYERS = model.config.num_hidden_layers 
model.requires_grad_(False)

# --- 3. PATCHING LOOP (With Hook Restoration) ---

from accelerate.hooks import add_hook_to_module

for i in range(NUM_LAYERS):
    layer_device = model.model.layers[i].mlp.gate.weight.device
    print(f"Patching Layer {i}/{NUM_LAYERS} on {layer_device}...")
    
    # 1. Capture the original expert and its hook
    old_expert = model.model.layers[i].mlp.shared_expert
    
    # 2. Create the new expert
    new_shared = LoRAAQExpert(HIDDEN_SIZE, r=128).to(layer_device)
    load_aq_weights_into_expert(new_shared, "./quant_weights", layer_idx=i)
    
    # 3. THE CRITICAL FIX: Restore the Accelerate Hook
    # This ensures the data actually moves to the correct GPU for your expert
    if hasattr(old_expert, "_hf_hook"):
        add_hook_to_module(new_shared, old_expert._hf_hook)
    
    # 4. Plug it back into the model
    model.model.layers[i].mlp.shared_expert = new_shared

torch.cuda.empty_cache()
gc.collect()

# --- 4. HEALING PREP ---

general_calibration = [
    """Quantum mechanics is a fundamental theory in physics describing nature at the scale of atoms. 
    The Industrial Revolution transformed manufacturing processes in Europe and the US. 
    The Ichimoku Cloud is a collection of technical indicators that show support and resistance levels. 
    def calculate_rsi(prices, window=9):
        delta = prices.diff()
        return 100 - (100 / (1 + (delta.where(delta > 0, 0).mean() / delta.where(delta < 0, 0).abs().mean())))
    Moral philosophy involves systematizing and recommending concepts of right and wrong conduct."""
]

trainable_params = []
for i in range(NUM_LAYERS):
    expert = model.model.layers[i].mlp.shared_expert
    for name, param in expert.named_parameters():
        # ONLY target the FP32 LoRA adapters. 
        if "lora" in name:
            param.requires_grad = True
            trainable_params.append(param)

optimizer = torch.optim.AdamW(trainable_params, lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# THE FIX: Ensure these three lines are complete and not truncated
model.gradient_checkpointing_disable()
model.config.use_cache = False 
model.train()

print(f"\n--- HEALING {len(trainable_params)} PARAMETERS ---")

# --- 5. HEALING LOOP ---

for epoch in range(15): 
    total_loss = 0
    for text in general_calibration:
        # Get the start device (cuda:0)
        start_device = next(model.parameters()).device
        batch = tokenizer(text, return_tensors="pt").to(start_device)
        
        # Explicit forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"].clone(),
            return_dict=True
        )
        
        loss = outputs.loss
        if loss is None:
            continue

        loss.backward()
        
        # Standard optimization step
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
    scheduler.step()
    # Clean print line
    avg_loss = total_loss / len(general_calibration)
    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Grad: {grad_norm:.2e}")

# --- 6. THE INFERENCE RESET ---

print("\n--- KILLING CHECKPOINTING GHOSTS ---")
model.eval()
model.config.use_cache = True
model.config.gradient_checkpointing = False

torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

for layer in model.model.layers:
    layer.gradient_checkpointing = False
    if hasattr(layer, "mlp"):
        layer.mlp.gradient_checkpointing = False

del optimizer; del trainable_params; gc.collect(); torch.cuda.empty_cache()

# --- 7. MMLU-PRO EVALUATION ---

print("\n--- STARTING MMLU-PRO (Nitro Mode) ---")

eval_model = HFLM(
    pretrained=model, 
    tokenizer=tokenizer, 
    batch_size=32
)

results = lm_eval.simple_evaluate(
    model=eval_model,
    tasks=["mmlu_pro"],
    num_fewshot=5
)

print(f"\n--- MMLU-PRO RESULTS ---")

# The fix: Ensure the dictionary lookup and print are on separate, clean lines
overall_acc = results['results']['mmlu_pro']['acc,none']
print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
