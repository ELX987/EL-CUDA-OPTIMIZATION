#!/usr/bin/env python3
"""
pretrain_ternary_llm.py

Pretraining scaffold for dense or MoE W1.58A8 ternary LLMs using EL_ternCUDA_kernel.cu.

It provides:
  - JSON / JSONL, Markdown, Parquet / .pq ingestion.
  - Hugging Face dataset fetching/streaming, including chat/conversation schema auto-detection.
  - Optional optimized JSON/Parquet/HF column with pre-tokenized token IDs.
  - Fast batched tokenization into a contiguous uint32 memmap cache.
  - Dense or MoE decoder-only Transformer with BitLinear layers.
  - Custom CUDA BitLinear calls through a small PyTorch C++ extension wrapper.
  - FP32 trainable/shadow weights, STE backward, W1.58 ternary packed safetensors export.
  - PyTorch SDPA/FlashAttention-compatible attention, optional torch.compile, DDP/FSDP.
  - Two-stage ternary QAT schedule: high LR/weight decay then cooldown with zero weight decay.

Important: tokenization produces integer token IDs. The 1.58-bit ternary representation is for
BitLinear weights; activations are quantized to INT8 by the custom CUDA kernel.

Dense local-data example:
  python pretrain_ternary_llm.py \
    --data 'data/*.jsonl' 'data/**/*.md' 'data/*.parquet' \
    --tokenizer ./tokenizer --output-dir runs/ternary_dense \
    --architecture dense --target-params 350M --layers 24 --seq-len 2048 \
    --batch-size 2 --grad-accum-steps 16 --max-steps 200000 \
    --kernel-cu ./EL_ternCUDA_kernel.cu --kernel-header ./EL_ternCUDA_kernel.h \
    --use-custom-kernel --compile


HF streaming example for the GLM-5.1 reasoning dataset:
  python pretrain_ternary_llm.py \
    --hf-dataset Jackrong/GLM-5.1-Reasoning-1M-Cleaned --hf-split train \
    --hf-streaming --stream-train --tokenizer auto \
    --output-dir runs/ternary_hf_stream --architecture dense --target-params 350M \
    --layers 24 --seq-len 2048 --batch-size 2 --grad-accum-steps 16 --max-tokens 10B \
    --kernel-cu ./EL_ternCUDA_kernel.cu --kernel-header ./EL_ternCUDA_kernel.h

MoE example:
  python pretrain_ternary_llm.py \
    --data 'data/train.parquet' --text-column text \
    --tokenizer ./tokenizer --output-dir runs/ternary_moe \
    --architecture moe --layers 24 --hidden-size 2048 --heads 16 \
    --num-experts 16 --top-k 2 --moe-num-layers 12 --moe-layer-stride 2 \
    --batch-size 1 --grad-accum-steps 32 --max-steps 200000 \
    --kernel-cu ./EL_ternCUDA_kernel.cu --kernel-header ./EL_ternCUDA_kernel.h \
    --use-custom-kernel
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import subprocess
import threading
import time
from functools import partial
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterator, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

EL_TERNARY_BUILD_VERSION = "2026-05-01-perf-fa4-ddp-flce-sonicrouting-dxcache-ddpstaticfix-fa4bs256-toolchainfix-v14"

try:
    from safetensors.torch import save_file as safe_save_file
except Exception:  # runtime checked
    safe_save_file = None


# =============================================================================
# Extension/source versioning
# =============================================================================

EL_TERNARY_TRAINER_VERSION = "2026-05-01.perf-fa4-ddp-flce-sonicrouting-dxcache.ddpstaticfix.fa4bs256.toolchainfix.v14"
# Bump this when the generated C++ binding or CUDA build assumptions change.
EL_TERNARY_EXTENSION_ABI_TAG = "headerless_streamhandle_v10_dxcache_fa4_flce_sonic_cudaaware"


# =============================================================================
# Config
# =============================================================================


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int = 2048
    architecture: str = "dense"  # dense | moe
    target_params: Optional[int] = None
    hidden_size: Optional[int] = None
    layers: int = 24
    heads: Optional[int] = None
    mlp_ratio: float = 4.0
    intermediate_size: Optional[int] = None
    multiple_of: int = 64
    activation: str = "relu2"  # relu2 | swiglu
    tie_embeddings: bool = True
    ternarize_lm_head: bool = False
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    gradient_checkpointing: bool = False

    # MoE
    num_experts: int = 8
    top_k: int = 2
    moe_num_layers: int = 0
    moe_layer_stride: int = 1
    moe_pad_to_multiple: int = 16
    router_aux_loss_coef: float = 0.01

    # Runtime model path
    use_custom_kernel: bool = True
    allow_torch_fallback: bool = False
    # Faster training path: custom ternary kernel for forward + dX, cached packed weights,
    # and torch.matmul for the STE dW = dY^T @ X gradient.
    # This avoids requiring CUDA library development headers in the custom extension build.
    # The flag name is retained for CLI compatibility.
    use_torch_dweight_grad: bool = True
    # Attention backend. flex-fa4 uses PyTorch FlexAttention with kernel_options={"BACKEND": "FLASH"}
    # when available, otherwise the script falls back to SDPA.
    attention_backend: str = "sdpa"  # sdpa | flex | flex-fa4
    flex_attention_compile: bool = True
    flex_block_size: int = 256
    # Backward math policy. On Blackwell, Tensor-Core dX is often faster than the
    # custom scalar ternary dX kernel, while the custom dX path remains available.
    bitlinear_dx_grad: str = "torch"  # torch | custom | custom-quantized
    bitlinear_dweight_dtype: str = "fp16"  # fp32/tf32 | bf16 | fp16
    # MoE routing. "sonic" is a local SonicMoE-style grouped router with optional
    # tile-aware token rounding; it does not copy or require SonicMoE kernels.
    moe_routing_backend: str = "grouped"  # naive | grouped | sonic
    moe_topk_over_softmax: bool = False
    moe_norm_topk_probs: bool = True
    moe_token_rounding: bool = False
    moe_token_rounding_strategy: str = "pad"  # pad | none


@dataclass
class TrainConfig:
    # Local data sources. May be empty when --hf-dataset is used.
    data: List[str] = dataclasses.field(default_factory=list)
    tokenizer: str = "auto"
    output_dir: str = "runs/ternary_pretrain"
    data_format: str = "auto"  # auto | json | markdown | parquet | paraquet
    text_column: str = "text"
    token_column: Optional[str] = None
    token_cache: Optional[str] = None
    rebuild_token_cache: bool = False
    append_eos: bool = True
    tokenization_batch_size: int = 512
    parquet_batch_size: int = 4096
    token_cache_max_tokens: Optional[int] = None
    vocab_size_override: Optional[int] = None

    # Hugging Face Hub dataset streaming/fetching.
    hf_dataset: List[str] = dataclasses.field(default_factory=list)
    hf_config: Optional[str] = None
    hf_split: str = "train"
    hf_streaming: bool = False
    stream_train: bool = False
    hf_data_files: Optional[List[str]] = None
    hf_cache_dir: Optional[str] = None
    hf_revision: Optional[str] = None
    hf_token_env: str = "HF_TOKEN"
    hf_trust_remote_code: bool = False
    hf_shuffle_buffer: int = 0
    hf_skip: int = 0
    hf_take: Optional[int] = None
    hf_interleave_probabilities: Optional[str] = None
    # DDP streaming sharding strategy for Hugging Face IterableDataset.
    # auto: use efficient shard/split only when there are enough HF data shards; otherwise rank-stride examples.
    # stride: every rank reads the same stream but keeps idx % world == rank; safe for single-shard streaming datasets.
    # shard/split: force HF shard or split_dataset_by_node; fastest when dataset has >= world shards.
    # none: every rank sees all samples (debug only; duplicates data across ranks).
    hf_ddp_shard_strategy: str = "stride"

    # Tokenizer resolution. The old example placeholder is treated like "auto".
    auto_tokenizer_fallback: str = "gpt2"
    tokenizer_cache_dir: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tokenizer_token_env: Optional[str] = None
    tokenizer_trust_remote_code: bool = False

    max_steps: int = 1000
    max_tokens: Optional[int] = None
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    cooldown_lr: Optional[float] = None
    warmup_steps: int = 200
    cooldown_start_frac: float = 0.5
    weight_decay_peak: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    optimizer: str = "adamw"  # adamw | adam8bit
    dtype: str = "fp16"  # runtime activation dtype; trainable parameters stay FP32 by default
    seed: int = 1337

    # Custom kernel / stack
    use_custom_kernel: bool = True
    allow_torch_fallback: bool = False
    use_torch_dweight_grad: bool = True
    bitlinear_dx_grad: str = "torch"
    bitlinear_dweight_dtype: str = "fp16"
    attention_backend: str = "sdpa"
    flex_attention_compile: bool = True
    flex_block_size: int = 256
    kernel_cu: str = "EL_ternCUDA_kernel.cu"
    kernel_header: str = "EL_ternCUDA_kernel.h"
    cuda_arch: Optional[str] = "auto"
    extension_verbose: bool = False
    extension_progress: bool = True
    extension_progress_interval: float = 2.0
    extension_build_dir: Optional[str] = None
    clean_extension_build: bool = False
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    use_flash_sdpa: bool = True
    use_liger: bool = True
    use_liger_fused_ce: bool = True
    fsdp: bool = False
    distributed: bool = False
    ddp_find_unused_parameters: str = "auto"  # auto | true | false
    ddp_static_graph: bool = False
    ddp_no_sync: bool = True
    ddp_gradient_as_bucket_view: bool = True

    log_interval: int = 10
    save_interval: int = 1000


# =============================================================================
# Basic utils
# =============================================================================


def parse_count(x: Optional[Union[str, int]]) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip().lower().replace("_", "")
    mult = 1
    for suffix, m in (("k", 10**3), ("m", 10**6), ("b", 10**9), ("t", 10**12)):
        if s.endswith(suffix):
            mult = m
            s = s[:-1]
            break
    return int(float(s) * mult)


def parse_optional_bool(value: Optional[Union[str, bool]]) -> bool:
    """Parse optional boolean CLI values for flags that may be passed as either
    --flag, --flag=true, --flag true, --flag 1, or omitted.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def human_int(n: Union[int, float]) -> str:
    v = float(n)
    for suffix in ("", "K", "M", "B", "T"):
        if abs(v) < 1000:
            return f"{v:.2f}{suffix}" if suffix else str(int(v))
        v /= 1000
    return f"{v:.2f}P"


def rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    if rank0():
        print(msg, flush=True)


def make_divisible(x: int, multiple: int) -> int:
    return int(math.ceil(int(x) / multiple) * multiple)


def expand_paths(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            p = Path(pat)
            if p.exists():
                out.append(p)
    files: List[Path] = []
    for p in out:
        if p.is_dir():
            for suffix in ("*.json", "*.jsonl", "*.md", "*.markdown", "*.parquet", "*.pq"):
                files.extend(p.rglob(suffix))
        elif p.is_file():
            files.append(p)
    seen = set()
    unique: List[Path] = []
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p.resolve())
    return unique


def infer_format(path: Path, requested: str) -> str:
    requested = requested.lower()
    if requested != "auto":
        return "parquet" if requested == "paraquet" else requested
    suf = path.suffix.lower()
    if suf in {".json", ".jsonl"}:
        return "json"
    if suf in {".md", ".markdown", ".mdown", ".mkd"}:
        return "markdown"
    if suf in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(f"Cannot infer data format for {path}; pass --data-format")


def init_distributed() -> Tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. The custom kernel path is CUDA-only.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1 and not torch.distributed.is_initialized():
        # Avoid NCCL "guessing device ID" warnings in newer PyTorch builds.
        try:
            torch.distributed.init_process_group(backend="nccl", device_id=device)
        except TypeError:
            torch.distributed.init_process_group(backend="nccl")
    return rank, world, local_rank, device


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    base = model
    changed = True
    while changed:
        changed = False
        if hasattr(base, "module"):
            base = base.module  # type: ignore[assignment]
            changed = True
        if hasattr(base, "_orig_mod"):
            base = base._orig_mod  # type: ignore[attr-defined,assignment]
            changed = True
    return base


# =============================================================================
# Data ingestion and token cache
# =============================================================================


def _coerce_token_ids(val: Any) -> Optional[List[int]]:
    """Return real token IDs only; token-count fields such as input_tokens are intentionally ignored."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            return [int(v) for v in val.tolist()]
        return None
    if isinstance(val, (list, tuple)):
        out: List[int] = []
        for v in val:
            if isinstance(v, (int, np.integer)):
                out.append(int(v))
            elif isinstance(v, float) and v.is_integer():
                out.append(int(v))
            else:
                return None
        return out
    if isinstance(val, str):
        parts = val.replace(",", " ").split()
        if parts and all(part.lstrip("-").isdigit() for part in parts):
            return [int(part) for part in parts]
    return None


def _message_text(msg: Any) -> Optional[str]:
    if isinstance(msg, str):
        return msg
    if not isinstance(msg, dict):
        return None
    for key in ("content", "value", "text", "message"):
        val = msg.get(key)
        if val is not None:
            if isinstance(val, list):
                chunks: List[str] = []
                for part in val:
                    if isinstance(part, str):
                        chunks.append(part)
                    elif isinstance(part, dict):
                        text = part.get("text") or part.get("content") or part.get("value")
                        if text is not None:
                            chunks.append(str(text))
                return "\n".join(chunks).strip() or None
            return str(val)
    return None


def _flatten_chat_messages(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    lines: List[str] = []
    for msg in messages:
        text = _message_text(msg)
        if not text:
            continue
        role = ""
        if isinstance(msg, dict):
            role = str(msg.get("role") or msg.get("from") or msg.get("speaker") or "").strip().lower()
        if role in {"human", "user", "question", "prompt"}:
            role = "user"
        elif role in {"gpt", "assistant", "model", "bot", "answer", "response"}:
            role = "assistant"
        elif role in {"system", "developer"}:
            role = role
        else:
            role = "message"
        lines.append(f"<|{role}|>\n{text}")
    return "\n".join(lines).strip() or None


def _join_prompt_response(obj: Dict[str, Any]) -> Optional[str]:
    pairs = [
        (("instruction", "prompt", "question", "query", "problem"), ("output", "response", "answer", "solution", "completion")),
        (("input",), ("output", "response", "answer", "completion")),
        (("prompt",), ("completion", "response", "answer")),
        (("question",), ("answer", "response", "solution")),
    ]
    for left_keys, right_keys in pairs:
        left_parts = [str(obj[k]) for k in left_keys if k in obj and obj[k] is not None and str(obj[k]).strip()]
        right_parts = [str(obj[k]) for k in right_keys if k in obj and obj[k] is not None and str(obj[k]).strip()]
        if left_parts or right_parts:
            out: List[str] = []
            if left_parts:
                out.append("\n".join(left_parts))
            if right_parts:
                out.append("\n".join(right_parts))
            return "\n\n".join(out).strip() or None
    return None


def _extract_json_item(obj: Any, text_col: str, token_col: Optional[str]) -> Union[str, List[int], None]:
    if isinstance(obj, dict):
        token_candidates = [token_col] if token_col else ["input_ids", "tokens", "token_ids", "ids"]
        for candidate in token_candidates:
            if candidate and candidate in obj:
                ids = _coerce_token_ids(obj[candidate])
                if ids is not None:
                    return ids

        if text_col in obj and obj[text_col] is not None:
            val = obj[text_col]
            if isinstance(val, list):
                chat = _flatten_chat_messages(val)
                if chat:
                    return chat
            return str(val)

        for chat_key in ("messages", "conversations", "conversation", "turns", "dialogue"):
            if chat_key in obj:
                chat = _flatten_chat_messages(obj[chat_key])
                if chat:
                    return chat

        joined = _join_prompt_response(obj)
        if joined:
            return joined

        # Fallback fields seen in web/HF corpora and instruction-tuning JSON.
        for k in ("text", "content", "markdown", "body", "document", "response", "output", "answer", "completion", "solution"):
            if k in obj and obj[k] is not None:
                return str(obj[k])
    if isinstance(obj, str):
        return obj
    return None

def iter_json(path: Path, text_col: str, token_col: Optional[str]) -> Iterator[Union[str, List[int]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = _extract_json_item(json.loads(line), text_col, token_col)
                except json.JSONDecodeError:
                    continue
                if item is not None:
                    yield item
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        obj = json.load(f)
    records = obj if isinstance(obj, list) else [obj]
    for rec in records:
        item = _extract_json_item(rec, text_col, token_col)
        if item is not None:
            yield item


def iter_markdown(path: Path) -> Iterator[str]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        yield text


def iter_parquet(path: Path, text_col: str, token_col: Optional[str], batch_size: int) -> Iterator[Union[str, List[int]]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet ingestion requires pyarrow: pip install pyarrow") from exc
    pf = pq.ParquetFile(path)
    schema_cols = set(pf.schema.names)
    if token_col is None:
        for cand in ("input_ids", "tokens", "token_ids", "ids"):
            if cand in schema_cols:
                token_col = cand
                break
    if token_col is None and text_col not in schema_cols:
        for cand in ("text", "content", "markdown", "body", "document"):
            if cand in schema_cols:
                text_col = cand
                break
    cols = [token_col] if token_col else [text_col]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        data = batch.to_pydict()
        if token_col:
            for val in data[token_col]:
                if val is None:
                    continue
                if isinstance(val, np.ndarray):
                    yield val.astype(np.int64).tolist()
                elif isinstance(val, list):
                    yield [int(x) for x in val]
                else:
                    yield [int(x) for x in str(val).replace(",", " ").split()]
        else:
            for val in data[text_col]:
                if val is not None:
                    yield str(val)


def iter_corpus(paths: Sequence[Path], cfg: TrainConfig) -> Iterator[Union[str, List[int]]]:
    for path in paths:
        fmt = infer_format(path, cfg.data_format)
        if fmt == "json":
            yield from iter_json(path, cfg.text_column, cfg.token_column)
        elif fmt == "markdown":
            yield from iter_markdown(path)
        elif fmt == "parquet":
            yield from iter_parquet(path, cfg.text_column, cfg.token_column, cfg.parquet_batch_size)
        else:
            raise ValueError(f"Unsupported data format: {fmt}")


_TOKENIZER_PLACEHOLDERS = {
    "",
    "auto",
    "./tokenizer_or_hf_tokenizer",
    "tokenizer_or_hf_tokenizer",
    "./tokenizer",
    "tokenizer",
    "path_or_hf_tokenizer",
    "./path_or_hf_tokenizer",
}


def _env_token(env_name: Optional[str]) -> Optional[str]:
    if not env_name:
        return None
    token = os.environ.get(env_name)
    return token if token else None


def _parse_hf_data_files(raw: Optional[List[str]]) -> Optional[Union[str, List[str], Dict[str, Any]]]:
    """Parse --hf-data-files values accepted by datasets.load_dataset."""
    if not raw:
        return None
    if len(raw) == 1:
        s = raw[0].strip()
        if not s:
            return None
        if s[0] in "[{":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
        if "=" in s and not s.startswith("hf://"):
            # train=foo.parquet,validation=bar.parquet
            out: Dict[str, Any] = {}
            for part in s.split(","):
                if not part.strip():
                    continue
                key, value = part.split("=", 1)
                out[key.strip()] = value.strip()
            return out
        return s
    if any("=" in item and not item.strip().startswith("hf://") for item in raw):
        out: Dict[str, List[str]] = {}
        for item in raw:
            key, value = item.split("=", 1)
            out.setdefault(key.strip(), []).append(value.strip())
        return {k: v[0] if len(v) == 1 else v for k, v in out.items()}
    return raw


def load_training_tokenizer(cfg: TrainConfig):
    """Load a tokenizer for text data, with a safe auto fallback for example placeholders."""
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Tokenization requires transformers and tokenizers") from exc

    requested = (cfg.tokenizer or "auto").strip()
    use_auto = requested in _TOKENIZER_PLACEHOLDERS or requested.lower() in _TOKENIZER_PLACEHOLDERS
    resolved = cfg.auto_tokenizer_fallback if use_auto else requested
    if use_auto:
        log(
            f"Tokenizer placeholder/auto value {requested!r} detected; "
            f"using --auto-tokenizer-fallback {resolved!r}."
        )

    kwargs: Dict[str, Any] = {
        "use_fast": True,
        "trust_remote_code": bool(cfg.tokenizer_trust_remote_code),
    }
    if cfg.tokenizer_cache_dir:
        kwargs["cache_dir"] = cfg.tokenizer_cache_dir
    if cfg.tokenizer_revision:
        kwargs["revision"] = cfg.tokenizer_revision
    token = _env_token(cfg.tokenizer_token_env or cfg.hf_token_env)
    if token:
        kwargs["token"] = token

    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved, **kwargs)
    except TypeError:
        # Older transformers/huggingface_hub used use_auth_token instead of token.
        if "token" in kwargs:
            tok = kwargs.pop("token")
            kwargs["use_auth_token"] = tok
        tokenizer = AutoTokenizer.from_pretrained(resolved, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Could not load tokenizer. The Hugging Face dataset passed via --hf-dataset is not used as a model/tokenizer. "
            "Pass a real tokenizer repo/path with --tokenizer, or use --tokenizer auto with "
            "--auto-tokenizer-fallback set to a real tokenizer such as gpt2 or your target model tokenizer. "
            f"Resolved tokenizer was {resolved!r}; original error: {exc}"
        ) from exc

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_one_hf_dataset(name: str, cfg: TrainConfig, *, streaming: bool):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Hugging Face dataset loading requires: pip install datasets huggingface_hub") from exc

    data_files = _parse_hf_data_files(cfg.hf_data_files)
    kwargs: Dict[str, Any] = {
        "split": cfg.hf_split,
        "streaming": bool(streaming),
        "trust_remote_code": bool(cfg.hf_trust_remote_code),
    }
    if cfg.hf_config:
        kwargs["name"] = cfg.hf_config
    if data_files is not None:
        kwargs["data_files"] = data_files
    if cfg.hf_cache_dir:
        kwargs["cache_dir"] = cfg.hf_cache_dir
    if cfg.hf_revision:
        kwargs["revision"] = cfg.hf_revision
    token = _env_token(cfg.hf_token_env)
    if token:
        kwargs["token"] = token

    try:
        return load_dataset(name, **kwargs)
    except TypeError:
        if "token" in kwargs:
            tok = kwargs.pop("token")
            kwargs["use_auth_token"] = tok
        return load_dataset(name, **kwargs)


def load_hf_dataset(cfg: TrainConfig, *, streaming: bool):
    if not cfg.hf_dataset:
        raise ValueError("No --hf-dataset supplied")
    datasets_list = [_load_one_hf_dataset(name, cfg, streaming=streaming) for name in cfg.hf_dataset]
    if len(datasets_list) == 1:
        ds = datasets_list[0]
    else:
        try:
            from datasets import interleave_datasets
        except Exception as exc:
            raise RuntimeError("Multiple --hf-dataset inputs require datasets.interleave_datasets") from exc
        probs = None
        if cfg.hf_interleave_probabilities:
            probs = [float(x) for x in cfg.hf_interleave_probabilities.split(",") if x.strip()]
            if len(probs) != len(datasets_list):
                raise ValueError("--hf-interleave-probabilities length must match the number of --hf-dataset entries")
        ds = interleave_datasets(datasets_list, probabilities=probs, seed=cfg.seed)
    if streaming and cfg.hf_shuffle_buffer and cfg.hf_shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=cfg.hf_shuffle_buffer, seed=cfg.seed)
    if cfg.hf_skip and cfg.hf_skip > 0:
        ds = ds.skip(cfg.hf_skip)
    if cfg.hf_take is not None:
        ds = ds.take(cfg.hf_take)
    return ds


def _distributed_shard_iterable(ds: Any, rank: int, world: int) -> Any:
    if world <= 1:
        return ds
    # Prefer HF's own sharding/splitting where available, but several Hub streaming
    # datasets expose fewer underlying data sources than the DDP world size. In that
    # case IterableDataset.shard can raise IndexError on nonzero ranks, so fall back
    # to rank-striding rows in Python.
    if hasattr(ds, "shard"):
        try:
            try:
                return ds.shard(num_shards=world, index=rank, contiguous=False)
            except TypeError:
                return ds.shard(num_shards=world, index=rank)
        except Exception as exc:
            if rank == 0:
                print(f"HF streaming dataset.shard failed ({type(exc).__name__}: {exc}); falling back to rank-stride DDP streaming.", flush=True)
    else:
        try:
            from datasets.distributed import split_dataset_by_node
            return split_dataset_by_node(ds, rank=rank, world_size=world)
        except Exception as exc:
            if rank == 0:
                print(f"HF streaming split_dataset_by_node failed ({type(exc).__name__}: {exc}); falling back to rank-stride DDP streaming.", flush=True)
    def gen():
        for i, row in enumerate(ds):
            if i % world == rank:
                yield row
    return gen()


def _tokens_from_value(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)):
                out.extend(_tokens_from_value(item))
            elif item is not None:
                out.append(int(item))
        return out
    if isinstance(value, str):
        # Handles strings like "1 2 3" or "1,2,3".
        return [int(x) for x in value.replace(",", " ").split() if x.strip()]
    return [int(value)]


def _stringify_messages(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            role = value.get("role") or value.get("from") or ""
            prefix = f"{role}: " if role else ""
            return prefix + str(value.get("content") or "")
        return "\n".join(f"{k}: {_stringify_messages(v)}" for k, v in value.items() if v is not None)
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify_messages(v) for v in value if v is not None)
    return str(value)


def _text_from_example(example: Dict[str, Any], cfg: TrainConfig) -> str:
    if cfg.text_column in example and example[cfg.text_column] is not None:
        return _stringify_messages(example[cfg.text_column])
    # Common HF/instruction/reasoning dataset fallbacks.
    for key in (
        "text", "content", "markdown", "body", "document", "response", "answer", "completion",
        "prompt", "question", "instruction", "messages", "conversations", "conversation",
    ):
        if key in example and example[key] is not None:
            return _stringify_messages(example[key])
    # Last resort: concatenate string-like fields while avoiding obvious metadata/id columns.
    parts: List[str] = []
    for key, value in example.items():
        lk = str(key).lower()
        if value is None or lk in {"id", "idx", "uuid", "source", "metadata"}:
            continue
        if isinstance(value, (str, dict, list, tuple)):
            text = _stringify_messages(value).strip()
            if text:
                parts.append(f"{key}: {text}")
    return "\n".join(parts)


def iter_hf_corpus(cfg: TrainConfig, *, streaming: bool, rank: int = 0, world: int = 1) -> Iterator[Union[str, List[int]]]:
    ds = load_hf_dataset(cfg, streaming=streaming)
    if streaming:
        ds = _distributed_shard_iterable(ds, rank, world)
    for example in ds:
        if cfg.token_column:
            value = example.get(cfg.token_column) if isinstance(example, dict) else None
            ids = _tokens_from_value(value)
            if ids:
                yield ids
        else:
            if not isinstance(example, dict):
                text = str(example)
            else:
                text = _text_from_example(example, cfg)
            if text:
                yield text


PLACEHOLDER_TOKENIZERS = {
    "", "auto", "./tokenizer", "./tokenizer_or_hf_tokenizer", "tokenizer_or_hf_tokenizer",
    "<tokenizer>", "<tokenizer_or_hf_tokenizer>", "./path/to/tokenizer",
}


def _looks_like_local_path(value: str) -> bool:
    return value.startswith("./") or value.startswith("../") or value.startswith("/") or value.endswith(".json")


def _dataset_hint_for_tokenizer(cfg: TrainConfig) -> Optional[str]:
    joined = " ".join(cfg.hf_dataset or [])
    low = joined.lower()
    if "glm-5.1" in low or "glm5.1" in low or "glm" in low:
        return "zai-org/GLM-5.1"
    if "qwen" in low:
        return "Qwen/Qwen3-Tokenizer"
    if "deepseek" in low:
        return "deepseek-ai/DeepSeek-V3.2"
    return None


def resolve_tokenizer_id(cfg: TrainConfig) -> str:
    requested = str(cfg.tokenizer or "auto").strip()
    if requested not in PLACEHOLDER_TOKENIZERS:
        if _looks_like_local_path(requested) and not Path(requested).exists():
            hint = _dataset_hint_for_tokenizer(cfg) or cfg.auto_tokenizer_fallback
            log(f"Tokenizer path {requested!r} does not exist; using {hint!r}. Override with --tokenizer <repo-or-local-path>.")
            return hint
        return requested
    hint = _dataset_hint_for_tokenizer(cfg) or cfg.auto_tokenizer_fallback
    log(f"Tokenizer set to {requested!r}; using {hint!r}. Override with --tokenizer <repo-or-local-path>.")
    return hint


def load_training_tokenizer(cfg: TrainConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Tokenization requires transformers and tokenizers: pip install transformers tokenizers") from exc
    tok_id = resolve_tokenizer_id(cfg)
    token_env = cfg.tokenizer_token_env or cfg.hf_token_env
    token = os.environ.get(token_env) if token_env else None
    kwargs: Dict[str, Any] = {"use_fast": True, "trust_remote_code": bool(cfg.tokenizer_trust_remote_code)}
    if cfg.tokenizer_cache_dir:
        kwargs["cache_dir"] = cfg.tokenizer_cache_dir
    if cfg.tokenizer_revision:
        kwargs["revision"] = cfg.tokenizer_revision
    if token:
        kwargs["token"] = token
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_id, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load tokenizer {tok_id!r}. The Hugging Face dataset passed via --hf-dataset is not a tokenizer/model. "
            "Pass a real tokenizer repo/local path, e.g. --tokenizer zai-org/GLM-5.1, "
            "--tokenizer Qwen/Qwen3-Tokenizer, or --tokenizer ./my_tokenizer."
        ) from exc
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _parse_hf_data_files(values: Optional[List[str]]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        raw = values[0]
        try:
            return json.loads(raw)
        except Exception:
            pass
        if "=" in raw:
            k, v = raw.split("=", 1)
            return {k: v}
        return raw
    split_map: Dict[str, List[str]] = {}
    loose: List[str] = []
    for raw in values:
        if "=" in raw:
            k, v = raw.split("=", 1)
            split_map.setdefault(k, []).append(v)
        else:
            loose.append(raw)
    if split_map and not loose:
        return {k: (v[0] if len(v) == 1 else v) for k, v in split_map.items()}
    if split_map and loose:
        split_map.setdefault("train", []).extend(loose)
        return split_map
    return loose


def _load_one_hf_dataset(name: str, cfg: TrainConfig, *, streaming: bool) -> Any:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Hugging Face loading requires datasets: pip install datasets huggingface_hub") from exc
    kwargs: Dict[str, Any] = {
        "split": cfg.hf_split,
        "streaming": bool(streaming),
        "trust_remote_code": bool(cfg.hf_trust_remote_code),
    }
    data_files = _parse_hf_data_files(cfg.hf_data_files)
    if data_files is not None:
        kwargs["data_files"] = data_files
    if cfg.hf_cache_dir:
        kwargs["cache_dir"] = cfg.hf_cache_dir
    if cfg.hf_revision:
        kwargs["revision"] = cfg.hf_revision
    token = os.environ.get(cfg.hf_token_env) if cfg.hf_token_env else None
    if token:
        kwargs["token"] = token
    if cfg.hf_config:
        return load_dataset(name, cfg.hf_config, **kwargs)
    return load_dataset(name, **kwargs)


def _parse_probabilities(s: Optional[str], n: int) -> Optional[List[float]]:
    if not s:
        return None
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"--hf-interleave-probabilities expected {n} values, got {len(vals)}")
    total = sum(vals)
    if total <= 0:
        raise ValueError("--hf-interleave-probabilities must sum to a positive value")
    return [v / total for v in vals]


def _hf_num_shards(ds: Any) -> Optional[int]:
    """Best-effort shard count for HF Dataset/IterableDataset objects."""
    for name in ("n_shards", "num_shards"):
        try:
            value = getattr(ds, name, None)
            if callable(value):
                value = value()
            if value is not None:
                return int(value)
        except Exception:
            pass
    return None


def _choose_hf_streaming_ddp_strategy(cfg: TrainConfig, ds: Any, world: int) -> str:
    strategy = (getattr(cfg, "hf_ddp_shard_strategy", "stride") or "stride").lower()
    if strategy not in {"auto", "split", "shard", "stride", "none"}:
        raise ValueError("--hf-ddp-shard-strategy must be one of auto, split, shard, stride, none")
    if world <= 1 or strategy == "none":
        return "none"
    if strategy in {"stride", "split", "shard"}:
        return strategy
    # Conservative default: rank-stride is slower than split/shard because each rank
    # walks the same remote stream, but it is robust. Hugging Face IterableDataset.shard()
    # and split_dataset_by_node() can fail lazily at iteration time with
    # IndexError("list index out of range") when a streaming dataset exposes fewer
    # underlying data sources than DDP ranks. The GLM-5.1 reasoning stream hit exactly
    # that failure on rank 1, so auto intentionally chooses stride unless the user
    # explicitly opts into --hf-ddp-shard-strategy split or shard.
    return "stride"


def _apply_hf_streaming_ddp_split(ds: Any, cfg: TrainConfig, *, rank: int, world: int) -> Tuple[Any, bool, str]:
    """Return (dataset, use_rank_stride, chosen_strategy).

    rank-stride is slower for remote streaming because every rank walks the same stream, but
    it is robust when the HF iterable has fewer underlying data shards than DDP ranks.
    """
    strategy = _choose_hf_streaming_ddp_strategy(cfg, ds, world)
    if world <= 1 or strategy == "none":
        return ds, False, strategy
    if strategy == "stride":
        return ds, True, strategy
    if strategy == "split":
        try:
            from datasets.distributed import split_dataset_by_node
            return split_dataset_by_node(ds, world_size=world, rank=rank), False, strategy
        except Exception as exc:
            if rank == 0:
                print(f"HF streaming split_dataset_by_node failed ({type(exc).__name__}: {exc}); falling back to rank-stride DDP streaming.", flush=True)
            return ds, True, "stride"
    if strategy == "shard":
        try:
            return ds.shard(num_shards=world, index=rank), False, strategy
        except Exception as exc:
            if rank == 0:
                print(f"HF streaming dataset.shard failed ({type(exc).__name__}: {exc}); falling back to rank-stride DDP streaming.", flush=True)
            return ds, True, "stride"
    return ds, True, "stride"


def load_hf_training_dataset(cfg: TrainConfig, *, streaming: Optional[bool] = None, rank: int = 0, world: int = 1, epoch: int = 0) -> Tuple[Any, bool, str]:
    if not cfg.hf_dataset:
        return None, False, "none"
    streaming = bool(cfg.hf_streaming or cfg.stream_train) if streaming is None else bool(streaming)
    loaded = [_load_one_hf_dataset(name, cfg, streaming=streaming) for name in cfg.hf_dataset]
    if len(loaded) == 1:
        ds = loaded[0]
    else:
        try:
            from datasets import interleave_datasets
        except Exception as exc:
            raise RuntimeError("Multiple --hf-dataset values require datasets.interleave_datasets") from exc
        ds = interleave_datasets(
            loaded,
            probabilities=_parse_probabilities(cfg.hf_interleave_probabilities, len(loaded)),
            seed=cfg.seed + epoch,
            stopping_strategy="first_exhausted",
        )

    use_rank_stride = False
    chosen_strategy = "none"
    if streaming and world > 1:
        ds, use_rank_stride, chosen_strategy = _apply_hf_streaming_ddp_split(ds, cfg, rank=rank, world=world)
        if rank == 0 and epoch == 0:
            n_shards = _hf_num_shards(ds)
            print(
                f"HF streaming DDP sharding: strategy={chosen_strategy}, world={world}, "
                f"rank_stride={use_rank_stride}, dataset_shards={n_shards if n_shards is not None else 'unknown'}",
                flush=True,
            )

    # Shuffle after the DDP split when a true shard/split is used. For rank-stride fallback,
    # every rank sees the same deterministic shuffled stream and keeps disjoint modulo indices.
    if cfg.hf_shuffle_buffer and hasattr(ds, "shuffle"):
        try:
            ds = ds.shuffle(buffer_size=int(cfg.hf_shuffle_buffer), seed=cfg.seed + epoch)
        except TypeError:
            ds = ds.shuffle(seed=cfg.seed + epoch)
    if cfg.hf_skip and hasattr(ds, "skip"):
        ds = ds.skip(int(cfg.hf_skip))
    if cfg.hf_take is not None and hasattr(ds, "take"):
        ds = ds.take(int(cfg.hf_take))
    return ds, use_rank_stride, chosen_strategy


def _yield_hf_records_with_optional_rank_stride(
    ds: Any,
    cfg: TrainConfig,
    *,
    rank: int,
    world: int,
    use_rank_stride: bool,
) -> Iterator[Union[str, List[int]]]:
    for stream_idx, rec in enumerate(ds):
        if use_rank_stride and world > 1 and (stream_idx % world) != rank:
            continue
        item = _extract_json_item(rec, cfg.text_column, cfg.token_column)
        if item is not None:
            yield item


def iter_hf_corpus(cfg: TrainConfig, *, rank: int = 0, world: int = 1, epoch: int = 0, streaming: Optional[bool] = None) -> Iterator[Union[str, List[int]]]:
    ds, use_rank_stride, strategy = load_hf_training_dataset(cfg, streaming=streaming, rank=rank, world=world, epoch=epoch)
    if ds is None:
        return
    try:
        yield from _yield_hf_records_with_optional_rank_stride(
            ds, cfg, rank=rank, world=world, use_rank_stride=use_rank_stride
        )
    except (IndexError, ValueError) as exc:
        # Some streaming datasets only fail once the iterator materializes the rank's
        # source list. Fall back to robust modulo/rank-stride mode instead of killing
        # the whole torchrun job.
        if streaming and world > 1 and strategy not in {"stride", "none"}:
            if rank == 0:
                print(
                    f"HF streaming DDP {strategy} failed during iteration "
                    f"({type(exc).__name__}: {exc}); retrying with rank-stride mode.",
                    flush=True,
                )
            old_strategy = cfg.hf_ddp_shard_strategy
            cfg.hf_ddp_shard_strategy = "stride"
            try:
                ds, use_rank_stride, strategy = load_hf_training_dataset(
                    cfg, streaming=streaming, rank=rank, world=world, epoch=epoch
                )
                yield from _yield_hf_records_with_optional_rank_stride(
                    ds, cfg, rank=rank, world=world, use_rank_stride=True
                )
            finally:
                cfg.hf_ddp_shard_strategy = old_strategy
        else:
            raise


def iter_all_corpus(cfg: TrainConfig, *, rank: int = 0, world: int = 1, epoch: int = 0, streaming: Optional[bool] = None) -> Iterator[Union[str, List[int]]]:
    paths = expand_paths(cfg.data)
    if paths:
        yield from iter_corpus(paths, cfg)
    if cfg.hf_dataset:
        yield from iter_hf_corpus(cfg, rank=rank, world=world, epoch=epoch, streaming=streaming)


def build_or_load_token_cache(cfg: TrainConfig) -> Tuple[Path, Dict[str, Any], Any]:
    tokenizer = load_training_tokenizer(cfg)
    cache_dir = Path(cfg.token_cache) if cfg.token_cache else Path(cfg.output_dir) / "token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = cache_dir / "tokens.uint32.bin"
    meta_path = cache_dir / "meta.json"

    if tokens_path.exists() and meta_path.exists() and not cfg.rebuild_token_cache:
        meta = json.loads(meta_path.read_text())
        log(f"Reusing token cache {tokens_path} ({human_int(meta['num_tokens'])} tokens)")
        return tokens_path, meta, tokenizer

    paths = expand_paths(cfg.data)
    if not paths and not cfg.hf_dataset:
        raise ValueError("No input data. Provide local --data paths and/or one or more --hf-dataset repos.")

    eos_id = tokenizer.eos_token_id
    append_eos = cfg.append_eos and eos_id is not None
    n_docs = 0
    n_tokens = 0
    max_tokens = int(cfg.token_cache_max_tokens) if cfg.token_cache_max_tokens is not None else None
    text_batch: List[str] = []
    started = time.time()

    def reached_limit() -> bool:
        return max_tokens is not None and n_tokens >= max_tokens

    def write_ids(f: io.BufferedWriter, ids: Sequence[int]) -> None:
        nonlocal n_tokens
        if not ids or reached_limit():
            return
        if max_tokens is not None:
            ids = ids[: max(0, max_tokens - n_tokens)]
        arr = np.asarray(ids, dtype=np.uint32)
        arr.tofile(f)
        n_tokens += int(arr.size)

    def flush(f: io.BufferedWriter) -> None:
        nonlocal text_batch
        if not text_batch or reached_limit():
            text_batch = []
            return
        enc = tokenizer(text_batch, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        for ids in enc["input_ids"]:
            if append_eos:
                ids = list(ids) + [int(eos_id)]
            write_ids(f, ids)
            if reached_limit():
                break
        text_batch = []

    sources = []
    if paths:
        sources.append(f"{len(paths)} local files")
    if cfg.hf_dataset:
        sources.append("HF:" + ",".join(cfg.hf_dataset))
    log(f"Building token cache at {tokens_path} from {' + '.join(sources)}")
    with tokens_path.open("wb") as f:
        for item in iter_all_corpus(cfg, streaming=cfg.hf_streaming):
            n_docs += 1
            if isinstance(item, list):
                flush(f)
                ids = list(item) + ([int(eos_id)] if append_eos else [])
                write_ids(f, ids)
            else:
                text_batch.append(item)
                if len(text_batch) >= cfg.tokenization_batch_size:
                    flush(f)
            if rank0() and n_docs % 10000 == 0:
                elapsed = max(1e-6, time.time() - started)
                log(f"  docs={n_docs:,} tokens={n_tokens:,} tok/s={n_tokens/elapsed:,.0f}")
            if reached_limit():
                break
        flush(f)

    if n_tokens < 2:
        raise RuntimeError("Token cache is empty; check --text-column/--token-column, HF dataset columns, and input files")
    meta = {
        "tokenizer": resolve_tokenizer_id(cfg),
        "vocab_size": int(cfg.vocab_size_override) if cfg.vocab_size_override else int(len(tokenizer)),
        "num_docs": int(n_docs),
        "num_tokens": int(n_tokens),
        "dtype": "uint32",
        "append_eos": bool(append_eos),
        "hf_dataset": list(cfg.hf_dataset),
        "created_unix": time.time(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    elapsed = max(1e-6, time.time() - started)
    log(f"Token cache complete: docs={n_docs:,} tokens={n_tokens:,} tok/s={n_tokens/elapsed:,.0f}")
    return tokens_path, meta, tokenizer


class TokenMemmap:
    def __init__(self, path: Union[str, Path], seq_len: int):
        self.path = Path(path)
        self.seq_len = int(seq_len)
        self.tokens = np.memmap(self.path, mode="r", dtype=np.uint32)
        if len(self.tokens) <= self.seq_len + 1:
            raise ValueError(f"Only {len(self.tokens)} tokens, too small for seq_len={seq_len}")

    def sample_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=batch_size, dtype=np.int64)
        x_np = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y_np = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for i, st in enumerate(starts):
            block = np.asarray(self.tokens[st: st + self.seq_len + 1], dtype=np.int64)
            x_np[i] = block[:-1]
            y_np[i] = block[1:]
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y

class StreamingTokenBatcher:
    """Sequential batcher for HF streaming mode; avoids writing a full token cache first."""

    def __init__(self, cfg: TrainConfig, tokenizer: Any, seq_len: int, *, rank: int = 0, world: int = 1):
        if not cfg.hf_dataset:
            raise ValueError("--stream-train currently requires --hf-dataset")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world = int(world)
        self.eos_id = tokenizer.eos_token_id
        self.append_eos = cfg.append_eos and self.eos_id is not None
        self.buffer: List[int] = []
        self.text_batch: List[str] = []
        self.iterator: Optional[Iterator[Union[str, List[int]]]] = None
        self.reset_iterator()

    def reset_iterator(self) -> None:
        self.iterator = iter_hf_corpus(self.cfg, streaming=True, rank=self.rank, world=self.world)

    def _append_ids(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        self.buffer.extend(int(x) for x in ids)
        if self.append_eos:
            self.buffer.append(int(self.eos_id))

    def _flush_text_batch(self) -> None:
        if not self.text_batch:
            return
        enc = self.tokenizer(
            self.text_batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for ids in enc["input_ids"]:
            self._append_ids(ids)
        self.text_batch = []

    def _pump_once(self) -> None:
        assert self.iterator is not None
        try:
            item = next(self.iterator)
        except StopIteration:
            self._flush_text_batch()
            self.reset_iterator()
            assert self.iterator is not None
            item = next(self.iterator)
        if isinstance(item, list):
            self._flush_text_batch()
            self._append_ids(item)
        else:
            self.text_batch.append(item)
            if len(self.text_batch) >= self.cfg.tokenization_batch_size:
                self._flush_text_batch()

    def _ensure_tokens(self, needed: int) -> None:
        while len(self.buffer) < needed:
            before = len(self.buffer)
            self._pump_once()
            # Avoid a dead loop if the dataset has many empty rows.
            if len(self.buffer) == before and self.text_batch:
                self._flush_text_batch()

    def sample_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        needed = int(batch_size) * (self.seq_len + 1)
        self._ensure_tokens(needed)
        x_np = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y_np = np.empty((batch_size, self.seq_len), dtype=np.int64)
        offset = 0
        for i in range(batch_size):
            block = self.buffer[offset: offset + self.seq_len + 1]
            x_np[i] = np.asarray(block[:-1], dtype=np.int64)
            y_np[i] = np.asarray(block[1:], dtype=np.int64)
            offset += self.seq_len + 1
        del self.buffer[:offset]
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y

class StreamingTokenBatcher:
    """Sequential direct-stream batcher for HF/local data without writing a token cache."""

    def __init__(self, cfg: TrainConfig, tokenizer: Any, seq_len: int, rank: int = 0, world: int = 1):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world = int(world)
        self.eos_id = tokenizer.eos_token_id
        self.append_eos = bool(cfg.append_eos and self.eos_id is not None)
        self.epoch = 0
        self.buffer: List[int] = []
        self.text_batch: List[str] = []
        self.docs_seen = 0
        self._reset_iterator()

    def _reset_iterator(self) -> None:
        self.iterator = iter_all_corpus(self.cfg, rank=self.rank, world=self.world, epoch=self.epoch, streaming=True)
        self.epoch += 1

    def _extend_ids(self, ids: Sequence[int]) -> None:
        self.buffer.extend(int(x) for x in ids)
        if self.append_eos:
            self.buffer.append(int(self.eos_id))

    def _flush_text_batch(self) -> None:
        if not self.text_batch:
            return
        enc = self.tokenizer(self.text_batch, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        for ids in enc["input_ids"]:
            self._extend_ids(ids)
        self.text_batch = []

    def _pull_docs(self, min_tokens: int) -> None:
        while len(self.buffer) < min_tokens:
            try:
                item = next(self.iterator)
            except StopIteration:
                self._flush_text_batch()
                self._reset_iterator()
                continue
            self.docs_seen += 1
            if isinstance(item, list):
                self._flush_text_batch()
                self._extend_ids(item)
            else:
                self.text_batch.append(item)
                if len(self.text_batch) >= self.cfg.tokenization_batch_size:
                    self._flush_text_batch()
        self._flush_text_batch()

    def sample_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        need = int(batch_size) * (self.seq_len + 1)
        self._pull_docs(need)
        arr = np.empty((batch_size, self.seq_len + 1), dtype=np.int64)
        offset = 0
        for b in range(batch_size):
            block = self.buffer[offset: offset + self.seq_len + 1]
            if len(block) < self.seq_len + 1:
                self._pull_docs(need)
                block = self.buffer[offset: offset + self.seq_len + 1]
            arr[b] = np.asarray(block, dtype=np.int64)
            offset += self.seq_len + 1
        del self.buffer[:offset]
        x = torch.from_numpy(arr[:, :-1].copy())
        y = torch.from_numpy(arr[:, 1:].copy())
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y


# =============================================================================
# Custom CUDA extension wrapper
# =============================================================================


_CPP_BINDING = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sstream>
#include <climits>
#include <cstdint>
#include "EL_ternCUDA_kernel.h"

#define EL_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define EL_CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define EL_CHECK_DTYPE(x, dtype) TORCH_CHECK((x).scalar_type() == dtype, #x " dtype mismatch")

static void check_err(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << where << " failed: " << cudaGetErrorString(err) << " (" << static_cast<int>(err) << ")";
        TORCH_CHECK(false, oss.str());
    }
}

static cudaStream_t el_stream_from_handle(torch::Tensor t, uint64_t stream_handle) {
    int dev = t.get_device();
    TORCH_CHECK(dev >= 0, "expected CUDA tensor with concrete device index");
    check_err(cudaSetDevice(dev), "cudaSetDevice");
    // The Python side passes torch.cuda.current_stream(...).cuda_stream as an integer.
    // This keeps kernels ordered with PyTorch work without including ATen/cuda/CUDAContext.h,
    // c10 CUDA stream headers or CUDA math-library development headers in the compiled extension.
    if (stream_handle == 0) {
        return nullptr;  // legacy default stream fallback
    }
    return reinterpret_cast<cudaStream_t>(stream_handle);
}


torch::Tensor bitlinear_forward_packed(torch::Tensor X, torch::Tensor W_packed, torch::Tensor W_scale, uint64_t stream_handle) {
    EL_CHECK_CUDA(X); EL_CHECK_CUDA(W_packed); EL_CHECK_CUDA(W_scale);
    EL_CHECK_CONTIG(X); EL_CHECK_CONTIG(W_packed); EL_CHECK_CONTIG(W_scale);
    EL_CHECK_DTYPE(X, at::ScalarType::Half);
    EL_CHECK_DTYPE(W_packed, at::ScalarType::Int);
    EL_CHECK_DTYPE(W_scale, at::ScalarType::Float);
    TORCH_CHECK(X.dim() == 2 && W_packed.dim() == 2 && W_scale.dim() == 1, "X [M,N], W_packed [K,words], W_scale [K] required");
    TORCH_CHECK(X.size(0) <= INT_MAX && X.size(1) <= INT_MAX && W_scale.size(0) <= INT_MAX, "kernel int32 shape limit exceeded");
    int M = static_cast<int>(X.size(0));
    int N = static_cast<int>(X.size(1));
    int K = static_cast<int>(W_scale.size(0));
    int expected_words = el_bitlinear_packed_words_per_row(N);
    TORCH_CHECK(W_packed.size(0) == K && W_packed.size(1) == expected_words, "packed weight shape mismatch");

    auto Y = torch::empty({X.size(0), W_scale.size(0)}, X.options());
    auto X_q = torch::empty({X.size(0), X.size(1)}, torch::TensorOptions().device(X.device()).dtype(torch::kInt8));
    auto X_scale = torch::empty({X.size(0)}, torch::TensorOptions().device(X.device()).dtype(torch::kFloat32));
    cudaStream_t stream = el_stream_from_handle(X, stream_handle);

    auto err = el_quantize_fp16_per_row_int8_async(
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<int8_t*>(X_q.data_ptr<int8_t>()),
        X_scale.data_ptr<float>(), M, N, stream);
    check_err(err, "el_quantize_fp16_per_row_int8_async");

    err = el_bitlinear_forward_prequantized_async(
        reinterpret_cast<const int8_t*>(X_q.data_ptr<int8_t>()),
        X_scale.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        W_scale.data_ptr<float>(),
        reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
        nullptr, M, N, K, stream);
    check_err(err, "el_bitlinear_forward_prequantized_async");
    return Y;
}

std::vector<torch::Tensor> bitlinear_forward_from_shadow_cached(torch::Tensor X, torch::Tensor W, uint64_t stream_handle) {
    EL_CHECK_CUDA(X); EL_CHECK_CUDA(W); EL_CHECK_CONTIG(X); EL_CHECK_CONTIG(W);
    EL_CHECK_DTYPE(X, at::ScalarType::Half); EL_CHECK_DTYPE(W, at::ScalarType::Float);
    TORCH_CHECK(X.dim() == 2 && W.dim() == 2, "X [M,N], W [K,N] required");
    TORCH_CHECK(X.size(1) == W.size(1), "X/W N mismatch");
    TORCH_CHECK(X.size(0) <= INT_MAX && X.size(1) <= INT_MAX && W.size(0) <= INT_MAX, "kernel int32 shape limit exceeded");
    int M = static_cast<int>(X.size(0));
    int N = static_cast<int>(X.size(1));
    int K = static_cast<int>(W.size(0));
    int words = el_bitlinear_packed_words_per_row(N);
    auto Y = torch::empty({X.size(0), W.size(0)}, X.options());
    auto packed = torch::empty({W.size(0), static_cast<int64_t>(words)}, torch::TensorOptions().device(W.device()).dtype(torch::kInt32));
    auto scale = torch::empty({W.size(0)}, torch::TensorOptions().device(W.device()).dtype(torch::kFloat32));
    auto X_q = torch::empty({X.size(0), X.size(1)}, torch::TensorOptions().device(X.device()).dtype(torch::kInt8));
    auto X_scale = torch::empty({X.size(0)}, torch::TensorOptions().device(X.device()).dtype(torch::kFloat32));
    cudaStream_t stream = el_stream_from_handle(X, stream_handle);

    auto err = el_pack_ternary_weights_async(W.data_ptr<float>(), reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), scale.data_ptr<float>(), K, N, stream);
    check_err(err, "el_pack_ternary_weights_async");
    err = el_quantize_fp16_per_row_int8_async(
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<int8_t*>(X_q.data_ptr<int8_t>()),
        X_scale.data_ptr<float>(), M, N, stream);
    check_err(err, "el_quantize_fp16_per_row_int8_async");
    err = el_bitlinear_forward_prequantized_async(
        reinterpret_cast<const int8_t*>(X_q.data_ptr<int8_t>()),
        X_scale.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
        nullptr, M, N, K, stream);
    check_err(err, "el_bitlinear_forward_prequantized_async");
    return {Y, packed, scale};
}

torch::Tensor bitlinear_forward_from_shadow(torch::Tensor X, torch::Tensor W, uint64_t stream_handle) {
    auto out = bitlinear_forward_from_shadow_cached(X, W, stream_handle);
    return out[0];
}

torch::Tensor bitlinear_backward_input_packed(torch::Tensor dY, torch::Tensor W_packed, torch::Tensor W_scale, int64_t in_features, uint64_t stream_handle) {
    EL_CHECK_CUDA(dY); EL_CHECK_CUDA(W_packed); EL_CHECK_CUDA(W_scale);
    EL_CHECK_CONTIG(dY); EL_CHECK_CONTIG(W_packed); EL_CHECK_CONTIG(W_scale);
    EL_CHECK_DTYPE(dY, at::ScalarType::Half);
    EL_CHECK_DTYPE(W_packed, at::ScalarType::Int);
    EL_CHECK_DTYPE(W_scale, at::ScalarType::Float);
    TORCH_CHECK(dY.dim() == 2 && W_packed.dim() == 2 && W_scale.dim() == 1, "dY [M,K], W_packed [K,words], W_scale [K] required");
    TORCH_CHECK(in_features > 0 && in_features <= INT_MAX, "invalid input feature count");
    TORCH_CHECK(dY.size(0) <= INT_MAX && dY.size(1) <= INT_MAX, "kernel int32 shape limit exceeded");
    int M = static_cast<int>(dY.size(0));
    int N = static_cast<int>(in_features);
    int K = static_cast<int>(dY.size(1));
    int expected_words = el_bitlinear_packed_words_per_row(N);
    TORCH_CHECK(W_scale.size(0) == K && W_packed.size(0) == K && W_packed.size(1) == expected_words, "packed weight/dY shape mismatch");
    auto dX = torch::empty({dY.size(0), in_features}, torch::TensorOptions().device(dY.device()).dtype(torch::kFloat16));
    cudaStream_t stream = el_stream_from_handle(dY, stream_handle);
    auto err = el_bitlinear_backward_input_async(
        reinterpret_cast<const __half*>(dY.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        W_scale.data_ptr<float>(),
        reinterpret_cast<__half*>(dX.data_ptr<at::Half>()),
        M, N, K, stream);
    check_err(err, "el_bitlinear_backward_input_async");
    return dX;
}

torch::Tensor bitlinear_backward_input_packed_quantized(torch::Tensor dY, torch::Tensor W_packed, torch::Tensor W_scale, int64_t in_features, uint64_t stream_handle) {
    EL_CHECK_CUDA(dY); EL_CHECK_CUDA(W_packed); EL_CHECK_CUDA(W_scale);
    EL_CHECK_CONTIG(dY); EL_CHECK_CONTIG(W_packed); EL_CHECK_CONTIG(W_scale);
    EL_CHECK_DTYPE(dY, at::ScalarType::Half);
    EL_CHECK_DTYPE(W_packed, at::ScalarType::Int);
    EL_CHECK_DTYPE(W_scale, at::ScalarType::Float);
    TORCH_CHECK(dY.dim() == 2 && W_packed.dim() == 2 && W_scale.dim() == 1, "dY [M,K], W_packed [K,words], W_scale [K] required");
    int M = static_cast<int>(dY.size(0));
    int N = static_cast<int>(in_features);
    int K = static_cast<int>(dY.size(1));
    int expected_words = el_bitlinear_packed_words_per_row(N);
    TORCH_CHECK(W_scale.size(0) == K && W_packed.size(0) == K && W_packed.size(1) == expected_words, "packed weight/dY shape mismatch");
    auto dX = torch::empty({dY.size(0), in_features}, torch::TensorOptions().device(dY.device()).dtype(torch::kFloat16));
    cudaStream_t stream = el_stream_from_handle(dY, stream_handle);
    auto err = el_bitlinear_backward_input_quantized_async(
        reinterpret_cast<const __half*>(dY.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        W_scale.data_ptr<float>(),
        reinterpret_cast<__half*>(dX.data_ptr<at::Half>()),
        M, N, K, stream);
    check_err(err, "el_bitlinear_backward_input_quantized_async");
    return dX;
}

torch::Tensor dequantize_packed_weights_half(torch::Tensor W_packed, torch::Tensor W_scale, int64_t in_features, uint64_t stream_handle) {
    EL_CHECK_CUDA(W_packed); EL_CHECK_CUDA(W_scale);
    EL_CHECK_CONTIG(W_packed); EL_CHECK_CONTIG(W_scale);
    EL_CHECK_DTYPE(W_packed, at::ScalarType::Int);
    EL_CHECK_DTYPE(W_scale, at::ScalarType::Float);
    TORCH_CHECK(in_features > 0 && in_features <= INT_MAX, "invalid input feature count");
    int K = static_cast<int>(W_scale.size(0));
    int N = static_cast<int>(in_features);
    int expected_words = el_bitlinear_packed_words_per_row(N);
    TORCH_CHECK(W_packed.size(0) == K && W_packed.size(1) == expected_words, "packed weight shape mismatch");
    auto W_deq = torch::empty({W_scale.size(0), in_features}, torch::TensorOptions().device(W_packed.device()).dtype(torch::kFloat16));
    cudaStream_t stream = el_stream_from_handle(W_packed, stream_handle);
    auto err = el_dequantize_packed_to_fp16_async(
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        W_scale.data_ptr<float>(),
        reinterpret_cast<__half*>(W_deq.data_ptr<at::Half>()),
        K, N, stream);
    check_err(err, "el_dequantize_packed_to_fp16_async");
    return W_deq;
}

std::vector<torch::Tensor> bitlinear_backward_packed(torch::Tensor X, torch::Tensor dY, torch::Tensor W_packed, torch::Tensor W_scale, uint64_t stream_handle) {
    EL_CHECK_CUDA(X); EL_CHECK_CUDA(dY); EL_CHECK_CUDA(W_packed); EL_CHECK_CUDA(W_scale);
    EL_CHECK_CONTIG(X); EL_CHECK_CONTIG(dY); EL_CHECK_CONTIG(W_packed); EL_CHECK_CONTIG(W_scale);
    EL_CHECK_DTYPE(X, at::ScalarType::Half); EL_CHECK_DTYPE(dY, at::ScalarType::Half);
    EL_CHECK_DTYPE(W_packed, at::ScalarType::Int); EL_CHECK_DTYPE(W_scale, at::ScalarType::Float);
    TORCH_CHECK(X.dim() == 2 && dY.dim() == 2, "X [M,N], dY [M,K] required");
    TORCH_CHECK(X.size(0) == dY.size(0), "X/dY M mismatch");
    int M = static_cast<int>(X.size(0));
    int N = static_cast<int>(X.size(1));
    int K = static_cast<int>(dY.size(1));
    auto dX = torch::empty_like(X);
    auto dW = torch::empty({dY.size(1), X.size(1)}, torch::TensorOptions().device(X.device()).dtype(torch::kFloat32));
    cudaStream_t stream = el_stream_from_handle(X, stream_handle);
    auto err = el_bitlinear_backward_async(
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dY.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        W_scale.data_ptr<float>(),
        reinterpret_cast<__half*>(dX.data_ptr<at::Half>()),
        dW.data_ptr<float>(), M, N, K, stream);
    check_err(err, "el_bitlinear_backward_async");
    return {dX, dW};
}

std::vector<torch::Tensor> bitlinear_backward_from_shadow(torch::Tensor X, torch::Tensor dY, torch::Tensor W, uint64_t stream_handle) {
    EL_CHECK_CUDA(X); EL_CHECK_CUDA(dY); EL_CHECK_CUDA(W);
    EL_CHECK_CONTIG(X); EL_CHECK_CONTIG(dY); EL_CHECK_CONTIG(W);
    EL_CHECK_DTYPE(X, at::ScalarType::Half); EL_CHECK_DTYPE(dY, at::ScalarType::Half); EL_CHECK_DTYPE(W, at::ScalarType::Float);
    TORCH_CHECK(X.dim() == 2 && dY.dim() == 2 && W.dim() == 2, "X [M,N], dY [M,K], W [K,N] required");
    TORCH_CHECK(X.size(0) == dY.size(0) && W.size(0) == dY.size(1) && X.size(1) == W.size(1), "shape mismatch");
    TORCH_CHECK(X.size(0) <= INT_MAX && X.size(1) <= INT_MAX && W.size(0) <= INT_MAX, "kernel int32 shape limit exceeded");
    int M = static_cast<int>(X.size(0));
    int N = static_cast<int>(X.size(1));
    int K = static_cast<int>(W.size(0));
    auto dX = torch::empty_like(X);
    auto dW = torch::empty_like(W);
    cudaStream_t stream = el_stream_from_handle(X, stream_handle);
    auto err = el_bitlinear_backward_from_shadow_async(
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dY.data_ptr<at::Half>()),
        W.data_ptr<float>(),
        reinterpret_cast<__half*>(dX.data_ptr<at::Half>()),
        dW.data_ptr<float>(), M, N, K, stream);
    check_err(err, "el_bitlinear_backward_from_shadow_async");
    return {dX, dW};
}


torch::Tensor dequantize_packed_to_fp16(torch::Tensor W_packed, torch::Tensor W_scale, int64_t in_features, uint64_t stream_handle) {
    EL_CHECK_CUDA(W_packed); EL_CHECK_CUDA(W_scale);
    EL_CHECK_CONTIG(W_packed); EL_CHECK_CONTIG(W_scale);
    EL_CHECK_DTYPE(W_packed, at::ScalarType::Int);
    EL_CHECK_DTYPE(W_scale, at::ScalarType::Float);
    TORCH_CHECK(W_packed.dim() == 2 && W_scale.dim() == 1, "W_packed [K,words], W_scale [K] required");
    TORCH_CHECK(in_features > 0 && in_features <= INT_MAX, "invalid input feature count");
    int K = static_cast<int>(W_scale.size(0));
    int N = static_cast<int>(in_features);
    int expected_words = el_bitlinear_packed_words_per_row(N);
    TORCH_CHECK(W_packed.size(0) == K && W_packed.size(1) == expected_words, "packed weight shape mismatch");
    auto W_deq = torch::empty({W_scale.size(0), in_features}, torch::TensorOptions().device(W_scale.device()).dtype(torch::kFloat16));
    cudaStream_t stream = el_stream_from_handle(W_packed, stream_handle);
    auto err = el_dequantize_packed_to_fp16_async(
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        W_scale.data_ptr<float>(),
        reinterpret_cast<__half*>(W_deq.data_ptr<at::Half>()),
        K, N, stream);
    check_err(err, "el_dequantize_packed_to_fp16_async");
    return W_deq;
}

std::vector<torch::Tensor> pack_ternary_weights(torch::Tensor W, uint64_t stream_handle) {
    EL_CHECK_CUDA(W); EL_CHECK_CONTIG(W); EL_CHECK_DTYPE(W, at::ScalarType::Float);
    TORCH_CHECK(W.dim() == 2, "W [K,N] required");
    TORCH_CHECK(W.size(0) <= INT_MAX && W.size(1) <= INT_MAX, "kernel int32 shape limit exceeded");
    int K = static_cast<int>(W.size(0));
    int N = static_cast<int>(W.size(1));
    int words = el_bitlinear_packed_words_per_row(N);
    auto packed = torch::empty({W.size(0), static_cast<int64_t>(words)}, torch::TensorOptions().device(W.device()).dtype(torch::kInt32));
    auto scale = torch::empty({W.size(0)}, torch::TensorOptions().device(W.device()).dtype(torch::kFloat32));
    cudaStream_t stream = el_stream_from_handle(W, stream_handle);
    auto err = el_pack_ternary_weights_async(W.data_ptr<float>(), reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), scale.data_ptr<float>(), K, N, stream);
    check_err(err, "el_pack_ternary_weights_async");
    return {packed, scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitlinear_forward_from_shadow", &bitlinear_forward_from_shadow);
    m.def("bitlinear_forward_from_shadow_cached", &bitlinear_forward_from_shadow_cached);
    m.def("bitlinear_forward_packed", &bitlinear_forward_packed);
    m.def("bitlinear_backward_from_shadow", &bitlinear_backward_from_shadow);
    m.def("bitlinear_backward_packed", &bitlinear_backward_packed);
    m.def("bitlinear_backward_input_packed", &bitlinear_backward_input_packed);
    m.def("bitlinear_backward_input_packed_quantized", &bitlinear_backward_input_packed_quantized);
    m.def("dequantize_packed_weights_half", &dequantize_packed_weights_half);
    m.def("pack_ternary_weights", &pack_ternary_weights);
    m.def("dequantize_packed_to_fp16", &dequantize_packed_to_fp16);
}
'''





def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:05.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _sha256_short(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def _write_text_if_changed(path: Path, text: str) -> bool:
    """Write only when content changed so PyTorch/JIT cache is not invalidated every launch."""
    if path.exists() and _safe_read_text(path) == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True

def _assert_extension_sources_are_current(kernel_cu: Path, binding_text: str) -> None:
    """Fail before ninja if an old source file would pull missing CUDA dev headers."""
    kernel_text = _safe_read_text(kernel_cu) if kernel_cu.exists() else ""
    problems: List[str] = []
    forbidden_kernel = [
        "#include <cublas_v2.h>",
        "#include <cusparse.h>",
        "cublasGemmEx",
        "cublasStatus_t",
    ]
    for needle in forbidden_kernel:
        if needle in kernel_text:
            problems.append(f"{kernel_cu} still contains `{needle}`")

    forbidden_binding = [
        "#include <ATen/cuda/CUDAContext.h>",
        "at::cuda::getCurrentCUDAStream",
        "-lcublas",
    ]
    for needle in forbidden_binding:
        if needle in binding_text:
            problems.append(f"generated binding still contains `{needle}`")

    if problems:
        details = "\n  - ".join(problems)
        raise RuntimeError(
            "The custom CUDA extension is about to compile stale/cuBLAS-dependent sources.\n"
            "This Vast/CUDA image is missing cuBLAS/cuSPARSE development headers, so the current trainer "
            "uses a header-light extension instead. Replace both /workspace/pretrain_ternary_llm.py and "
            "/workspace/EL_ternCUDA_kernel.cu with the latest files, then rebuild once.\n"
            f"Problems found:\n  - {details}"
        )


def _check_cuda_dev_headers_for_extension() -> None:
    """Fast preflight for the headers this extension actually includes.

    Vast.ai images sometimes provide nvcc and runtime libraries but omit CUDA math
    development headers. The current extension intentionally avoids cublas_v2.h and
    ATen/cuda/CUDAContext.h so it should only need cuda_runtime.h/cuda_fp16.h plus
    PyTorch C++ extension headers. This function prints a useful warning rather than failing
    silently during ninja.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    include_dir = Path(cuda_home) / "include"
    required = ["cuda_runtime.h", "cuda_fp16.h"]
    missing = [name for name in required if not (include_dir / name).exists()]
    if missing and rank0():
        log(f"WARNING: CUDA include directory {include_dir} is missing {missing}. "
            "This image may not contain CUDA development headers; install/use a CUDA devel image if compilation fails.")



def _validate_kernel_source_for_minimal_build(cu: Path) -> None:
    """Catch stale kernel sources before ninja fails with missing CUDA math headers."""
    text = _safe_read_text(cu)
    banned = [
        "#include <cublas_v2.h>",
        "#include <cusparse.h>",
        "#include <ATen/cuda/CUDAContext.h>",
        "#include <c10/cuda/CUDAGuard.h>",
    ]
    hits = [b for b in banned if b in text]
    if hits:
        raise RuntimeError(
            "The kernel source being compiled still contains stale CUDA/PyTorch headers "
            f"that are not available in minimal Vast.ai images: {hits}.\n"
            f"File checked: {cu}\n"
            "Replace /workspace/EL_ternCUDA_kernel.cu with the updated kernel from this chat, "
            "then rerun with --clean-extension-build once. Libtorch/PyTorch is already being "
            "used by torch.utils.cpp_extension; it cannot supply NVIDIA headers such as cublas_v2.h."
        )


def _preflight_extension_source_files(cu: Path, binding_cpp: Path) -> None:
    """Validate the exact kernel and generated binding that ninja will compile."""
    _assert_extension_sources_are_current(cu, _safe_read_text(binding_cpp))
    _validate_kernel_source_for_minimal_build(cu)


def _normalize_torch_cuda_arch(value: Optional[str]) -> Optional[str]:
    """Normalize common CUDA arch spellings to TORCH_CUDA_ARCH_LIST format.

    Examples:
      sm_100, compute_100, 100, 10.0 -> 10.0
      sm_120, 120, 12.0+PTX -> 12.0 or 12.0+PTX
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "auto":
        return None
    keep_ptx = "+ptx" in raw.lower()
    s = raw.lower().replace("+ptx", "")
    s = s.replace("sm_", "").replace("compute_", "")
    s = s.replace("sm", "") if s.startswith("sm") else s
    s = s.strip()
    if ";" in s or "," in s or " " in s:
        # Normalize only the first arch for build-dir tagging; full env strings are handled separately.
        for sep in (";", ",", " "):
            s = s.split(sep)[0]
        s = s.strip()
    if not s:
        return None
    if "." in s:
        parts = s.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    else:
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            return None
        # NVIDIA arch shorthand: 86 -> 8.6, 90 -> 9.0, 100 -> 10.0, 120 -> 12.0.
        if len(digits) <= 2:
            major = int(digits[0])
            minor = int(digits[1]) if len(digits) == 2 else 0
        else:
            major = int(digits[:-1])
            minor = int(digits[-1])
    out = f"{major}.{minor}"
    return out + "+PTX" if keep_ptx else out


def _split_torch_cuda_arch_list(value: str) -> List[str]:
    tokens: List[str] = []
    for chunk in value.replace(",", ";").replace(" ", ";").split(";"):
        norm = _normalize_torch_cuda_arch(chunk)
        if norm:
            tokens.append(norm)
    return tokens


def _current_cuda_arch() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (torch_arch, sm_tag, device_name) for the active CUDA device."""
    if not torch.cuda.is_available():
        return None, None, None
    try:
        device_index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
        name = torch.cuda.get_device_name(device_index)
        return f"{major}.{minor}", f"sm_{major}{minor}", name
    except Exception:
        return None, None, None


def _arch_to_build_suffix(arch: Optional[str]) -> str:
    if not arch:
        return "auto"
    tokens = _split_torch_cuda_arch_list(arch)
    first = tokens[0] if tokens else arch
    first = first.replace("+PTX", "")
    norm = _normalize_torch_cuda_arch(first) or first
    digits = norm.replace(".", "").replace("+PTX", "")
    return "sm" + digits


def _cuda_toolkit_build_suffix() -> Tuple[str, str]:
    """Return (suffix, human_readable_version) for extension cache partitioning.

    CUDA minor updates can change NVCC/PTX codegen and FA4/CuTe dependencies.
    Including the toolkit version in the build folder prevents loading a stale
    .so compiled under CUDA 13.0 after a CUDA 13.2 upgrade.
    """
    version = getattr(torch.version, "cuda", None) or "unknown"
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        try:
            proc = subprocess.run([nvcc_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5)
            m = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", proc.stdout or "")
            if m:
                version = m.group(1)
        except Exception:
            pass
    compact = re.sub(r"[^0-9]", "", str(version)) or "unknown"
    return f"cu{compact}", str(version)


def _cuda_toolchain_summary() -> Dict[str, str]:
    """Best-effort CUDA/PyTorch toolchain summary for extension build logs.

    This helper is intentionally non-fatal and display-only. It does not import
    CUDA math-library headers or use ATen CUDAContext, so it remains safe on
    cloud images with partial CUDA development packages.
    """
    torch_version = str(getattr(torch, "__version__", "unknown"))
    torch_cuda = str(getattr(torch.version, "cuda", None) or "unknown")

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
    try:
        from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME  # type: ignore
        if TORCH_CUDA_HOME:
            cuda_home = str(TORCH_CUDA_HOME)
    except Exception:
        pass
    if not cuda_home:
        cuda_home = "<unset>"

    nvcc_path = shutil.which("nvcc")
    if not nvcc_path and cuda_home and cuda_home != "<unset>":
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.exists():
            nvcc_path = str(candidate)

    nvcc_version = "nvcc not found"
    if nvcc_path:
        try:
            proc = subprocess.run(
                [nvcc_path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=8,
            )
            lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
            if lines:
                nvcc_version = lines[-1]
        except Exception as exc:
            nvcc_version = f"nvcc version unavailable: {type(exc).__name__}: {exc}"
    else:
        nvcc_path = "<not found>"

    return {
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "cuda_home": str(cuda_home),
        "nvcc_path": str(nvcc_path),
        "nvcc_version": str(nvcc_version),
    }


def _configure_cuda_arch_for_extension(cfg: TrainConfig) -> Dict[str, Optional[str]]:
    """Set TORCH_CUDA_ARCH_LIST to the active device unless the user explicitly overrides it.

    This prevents stale/incorrect JIT extension builds such as compiling sm_90 for a B200
    (which is sm_100 / compute capability 10.0), then failing at runtime with
    "no kernel image is available for execution on the device".
    """
    detected_arch, detected_sm, detected_name = _current_cuda_arch()
    requested_arch = _normalize_torch_cuda_arch(cfg.cuda_arch)
    existing_env = os.environ.get("TORCH_CUDA_ARCH_LIST")

    chosen = requested_arch or detected_arch
    reason = "--cuda-arch" if requested_arch else "auto-detected CUDA device"

    if chosen:
        if existing_env:
            existing_norm = _split_torch_cuda_arch_list(existing_env)
            chosen_no_ptx = (chosen or "").replace("+PTX", "")
            existing_no_ptx = [x.replace("+PTX", "") for x in existing_norm]
            if requested_arch:
                # CLI flag is explicit; it wins over the environment.
                if existing_env != requested_arch:
                    log(f"[cuda-build] overriding TORCH_CUDA_ARCH_LIST={existing_env!r} with {requested_arch!r} from --cuda-arch")
                os.environ["TORCH_CUDA_ARCH_LIST"] = requested_arch
            elif chosen_no_ptx not in existing_no_ptx or len(existing_no_ptx) != 1:
                log(
                    f"[cuda-build] overriding TORCH_CUDA_ARCH_LIST={existing_env!r} with {chosen!r} "
                    f"for detected device {detected_name or '<unknown>'} ({detected_sm or chosen})."
                )
                os.environ["TORCH_CUDA_ARCH_LIST"] = chosen
            # else env already exactly targets the active device; keep it.
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = chosen
    elif existing_env:
        chosen = existing_env
        reason = "existing TORCH_CUDA_ARCH_LIST"

    return {
        "detected_arch": detected_arch,
        "detected_sm": detected_sm,
        "detected_name": detected_name,
        "requested_arch": requested_arch,
        "chosen_arch": os.environ.get("TORCH_CUDA_ARCH_LIST", chosen),
        "reason": reason,
    }


def _smoke_test_custom_cuda_extension(ext: Any) -> None:
    """Fail fast with a clear message if the compiled cubin cannot run on this GPU."""
    if ext is None or not torch.cuda.is_available():
        return
    detected_arch, detected_sm, detected_name = _current_cuda_arch()
    try:
        # Small shapes that exercise pack, forward, and dX kernels without stressing memory.
        x = torch.zeros((2, 16), device="cuda", dtype=torch.float16)
        w = torch.zeros((3, 16), device="cuda", dtype=torch.float32)
        y, packed, scale = ext.bitlinear_forward_from_shadow_cached(x, w, _cuda_stream_handle(x))
        dy = torch.ones_like(y)
        dx = ext.bitlinear_backward_input_packed(dy, packed, scale, 16, _cuda_stream_handle(dy))
        if hasattr(ext, "bitlinear_backward_input_packed_quantized"):
            dxq = ext.bitlinear_backward_input_packed_quantized(dy, packed, scale, 16, _cuda_stream_handle(dy))
        else:
            dxq = None
        if hasattr(ext, "dequantize_packed_weights_half"):
            wdeq = ext.dequantize_packed_weights_half(packed, scale, 16, _cuda_stream_handle(dy))
        else:
            wdeq = None
        torch.cuda.synchronize()
        del x, w, y, packed, scale, dy, dx, dxq, wdeq
    except RuntimeError as exc:
        text = str(exc)
        if "no kernel image is available" in text or "invalid device function" in text:
            raise RuntimeError(
                "The custom CUDA extension loaded, but its kernels were not compiled for this GPU. "
                f"Detected device: {detected_name or '<unknown>'} ({detected_sm or detected_arch or '<unknown arch>'}). "
                "For an NVIDIA B200, use compute capability 10.0 / sm_100. Run:\n\n"
                "  export TORCH_CUDA_ARCH_LIST=10.0\n"
                "  python pretrain_ternary_llm.py ... --cuda-arch 10.0 --clean-extension-build\n\n"
                "Or delete the extension build directory printed above, then rerun. "
                "A stale sm_90/sm_89 build will load but fail at the first kernel launch."
            ) from exc
        raise


def _count_ninja_edges(build_dir: Path) -> int:
    build_ninja = build_dir / "build.ninja"
    if not build_ninja.exists():
        return 0
    total = 0
    for line in _safe_read_text(build_ninja).splitlines():
        line = line.strip()
        if not line.startswith("build "):
            continue
        # Count real compile/link edges, not phony aliases.
        if any(token in line for token in (".o:", ".so:", ".pyd:", ".dll:", ".dylib:")):
            total += 1
    return total


def _count_ninja_log_entries(build_dir: Path) -> int:
    ninja_log = build_dir / ".ninja_log"
    if not ninja_log.exists():
        return 0
    count = 0
    for line in _safe_read_text(ninja_log).splitlines():
        if not line or line.startswith("#"):
            continue
        count += 1
    return count


def _latest_build_artifact(build_dir: Path) -> str:
    try:
        candidates = [p for p in build_dir.iterdir() if p.is_file()]
    except Exception:
        return "none yet"
    if not candidates:
        return "none yet"
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name


def _detect_build_phase(build_dir: Path) -> str:
    try:
        files = {p.name for p in build_dir.iterdir() if p.is_file()}
    except Exception:
        files = set()
    if "build.ninja" not in files:
        return "preparing build.ninja"
    if any(name.endswith((".so", ".pyd", ".dll", ".dylib")) for name in files):
        return "link complete / loading"
    if any(name.endswith(".o") for name in files):
        return "compiling/linking"
    return "ninja build queued"


class _CudaExtensionBuildMonitor:
    """Small live monitor for torch.utils.cpp_extension.load(), which is otherwise blocking."""

    def __init__(self, build_dir: Path, interval: float, enabled: bool):
        self.build_dir = build_dir
        self.interval = max(0.5, float(interval))
        self.enabled = bool(enabled and rank0())
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._base_log_entries = _count_ninja_log_entries(build_dir)
        self._last_line_len = 0

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="cuda-extension-build-monitor", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval + 0.5))
        if self.enabled:
            elapsed = _format_duration(time.monotonic() - self.started)
            status = "failed" if exc_type is not None else "done"
            print(f"\r[cuda-build] {status} after {elapsed}. Build dir: {self.build_dir}" + " " * 20, flush=True)
        return False

    def _bar(self, done: int, total: int, width: int = 24) -> str:
        if total <= 0:
            # Indeterminate animated bar.
            tick = int((time.monotonic() - self.started) / self.interval) % width
            chars = ["-"] * width
            chars[tick] = ">"
            return "[" + "".join(chars) + "]"
        done = max(0, min(done, total))
        filled = int(round(width * done / max(1, total)))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _run(self) -> None:
        warned_slow = False
        while not self._stop.wait(self.interval):
            elapsed = time.monotonic() - self.started
            total = _count_ninja_edges(self.build_dir)
            current_entries = _count_ninja_log_entries(self.build_dir)
            done = max(0, current_entries - self._base_log_entries)
            phase = _detect_build_phase(self.build_dir)
            latest = _latest_build_artifact(self.build_dir)
            bar = self._bar(done, total)
            progress = f"{done}/{total}" if total else "?/ ?"
            line = f"[cuda-build] {bar} {progress} | elapsed {_format_duration(elapsed)} | {phase} | latest: {latest}"
            # Print full lines instead of carriage-return-only updates so logs on Vast/cloud consoles preserve history.
            print(line, flush=True)
            if elapsed > 180 and not warned_slow:
                warned_slow = True
                print(
                    "[cuda-build] still building after 3 minutes. Common causes: compiling for many GPU arches, "
                    "low MAX_JOBS, first-run ninja setup, or stale/locked build files. Consider setting "
                    "TORCH_CUDA_ARCH_LIST to your GPU only and using --clean-extension-build once.",
                    flush=True,
                )

def _distributed_barrier_if_ready() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            if torch.cuda.is_available():
                torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            else:
                torch.distributed.barrier()
        except TypeError:
            torch.distributed.barrier()


def _compile_or_load_custom_cuda_extension(cfg: TrainConfig):
    cu = Path(cfg.kernel_cu).resolve()
    header = Path(cfg.kernel_header).resolve()
    if not cu.exists() or not header.exists():
        msg = f"Kernel files not found: {cu} / {header}"
        if cfg.allow_torch_fallback:
            log(msg + "; falling back to PyTorch STE")
            return None
        raise FileNotFoundError(msg)
    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:
        if cfg.allow_torch_fallback:
            log(f"Cannot import torch cpp_extension ({exc}); falling back")
            return None
        raise

    _assert_extension_sources_are_current(cu, _CPP_BINDING)

    arch_info = _configure_cuda_arch_for_extension(cfg)
    chosen_arch = arch_info.get("chosen_arch")
    arch_suffix = _arch_to_build_suffix(chosen_arch)
    cuda_suffix, cuda_toolkit_version = _cuda_toolkit_build_suffix()
    build_dir = (
        Path(cfg.extension_build_dir).expanduser()
        if cfg.extension_build_dir
        else Path(cfg.output_dir) / f"cuda_extension_build_{EL_TERNARY_EXTENSION_ABI_TAG}_{arch_suffix}_{cuda_suffix}"
    )
    build_dir = build_dir.resolve()
    _check_cuda_dev_headers_for_extension()
    if cfg.clean_extension_build and build_dir.exists():
        log(f"Removing existing CUDA extension build directory: {build_dir}")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    binding_cpp = build_dir / "eltern_torch_bindings.cpp"
    wrote_binding = _write_text_if_changed(binding_cpp, _CPP_BINDING)
    _preflight_extension_source_files(cu, binding_cpp)

    # Do not pass -arch directly here. torch.utils.cpp_extension.load generates the
    # correct -gencode flags from TORCH_CUDA_ARCH_LIST, which we normalize above.
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo"]

    if rank0():
        arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "<unset; PyTorch auto-detects visible GPU arch>")
        max_jobs = os.environ.get("MAX_JOBS", "<unset; ninja chooses>")
        log(f"EL ternary trainer version: {EL_TERNARY_TRAINER_VERSION}")
        log(f"EL ternary extension ABI tag: {EL_TERNARY_EXTENSION_ABI_TAG}")
        log("Building/loading EL ternary CUDA extension...")
        log(f"  extension_source_version: {EL_TERNARY_BUILD_VERSION}")
        log(f"  build_directory: {build_dir}")
        log(f"  kernel_cu:       {cu} sha256={_sha256_short(cu)}")
        log(f"  kernel_header:   {header} sha256={_sha256_short(header)}")
        log(f"  binding_cpp:     {binding_cpp} sha256={_sha256_short(binding_cpp)} ({'rewritten' if wrote_binding else 'unchanged/cache-friendly'})")
        log(f"  launcher_script: {Path(__file__).resolve()} sha256={_sha256_short(Path(__file__).resolve())}")
        log(f"  detected_device: {arch_info.get('detected_name') or '<unknown>'} ({arch_info.get('detected_sm') or '<unknown sm>'})")
        log(f"  cuda_arch_source: {arch_info.get('reason')} | chosen_arch={arch_info.get('chosen_arch')}")
        log(f"  TORCH_CUDA_ARCH_LIST={arch_env} | MAX_JOBS={max_jobs}")
        _toolchain = _cuda_toolchain_summary()
        log(f"  torch={_toolchain['torch_version']} | torch.version.cuda={_toolchain['torch_cuda']} | CUDA_HOME={_toolchain['cuda_home']}")
        log(f"  nvcc={_toolchain['nvcc_path']} | {_toolchain['nvcc_version']}")
        log("  extension ABI: nocublas_v2; cached FP16 dX matmul + optional packed/quantized dX; dW uses torch.matmul")
        if not cfg.extension_verbose:
            log("  tip: add --extension-verbose for full ninja/nvcc command output")

    try:
        with _CudaExtensionBuildMonitor(build_dir, cfg.extension_progress_interval, cfg.extension_progress):
            ext = load(
                name=f"eltern_cuda_ext_{arch_suffix}_{cuda_suffix}_headerless_streamhandle_v10_dxcache",
                sources=[str(binding_cpp), str(cu)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=[str(header.parent)],
                with_cuda=True,
                verbose=bool(cfg.extension_verbose),
                build_directory=str(build_dir),
            )
        _smoke_test_custom_cuda_extension(ext)
        log("Custom CUDA extension loaded and smoke-tested on the active GPU.")
        return ext
    except Exception as exc:
        if cfg.allow_torch_fallback:
            log(f"Custom CUDA extension failed: {exc}\nFalling back to PyTorch STE.")
            return None
        raise


def load_custom_cuda_extension(cfg: TrainConfig):
    if not cfg.use_custom_kernel:
        log("Custom kernel disabled; using PyTorch STE fallback.")
        return None

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    # Avoid every rank compiling the same extension at once. Rank 0 builds first;
    # other ranks wait, then load the already-built shared object from disk.
    if world > 1 and rank != 0:
        if bool(cfg.extension_progress):
            print("[cuda-build] nonzero rank waiting for rank 0 extension build...", flush=True)
        _distributed_barrier_if_ready()
        rank_cfg = dataclasses.replace(cfg, clean_extension_build=False, extension_progress=False)
        return _compile_or_load_custom_cuda_extension(rank_cfg)

    ext = _compile_or_load_custom_cuda_extension(cfg)
    if world > 1 and rank == 0:
        _distributed_barrier_if_ready()
    return ext

# =============================================================================
# BitLinear and model components
# =============================================================================


def fake_quant_act_int8_ste(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    q = torch.round(x / scale).clamp(-128, 127)
    return x + (q * scale - x).detach()


def ternary_weight_ste(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scale = w.detach().abs().mean(dim=1, keepdim=True).clamp_min(eps)
    q = torch.round(w / scale).clamp(-1, 1)
    return w + (q * scale - w).detach()


def ternary_weight_deq_detached(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scale = w.detach().abs().mean(dim=1, keepdim=True).clamp_min(eps)
    q = torch.round(w.detach() / scale).clamp(-1, 1)
    return q * scale


def _cuda_stream_handle(t: torch.Tensor) -> int:
    if not t.is_cuda:
        return 0
    return int(torch.cuda.current_stream(t.device).cuda_stream)




def _ste_dweight_matmul(dy: torch.Tensor, x: torch.Tensor, precision: str = "fp16") -> torch.Tensor:
    """STE dW = dY^T @ X using a selectable input precision and FP32 output."""
    precision = str(precision or "fp16").lower()
    if precision == "fp32":
        return dy.float().transpose(0, 1).matmul(x.float())
    if precision == "bf16":
        return dy.to(torch.bfloat16).transpose(0, 1).matmul(x.to(torch.bfloat16)).float()
    # Default: half input GEMM with FP32 output. This is usually fastest on B200.
    return dy.to(torch.float16).transpose(0, 1).matmul(x.to(torch.float16)).float()

def _dequantized_dx_matmul(
    kernel_ext: Any,
    dY2d: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    in_features: int,
    w_shadow: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """dX = dY @ dequant(W_ternary), using extension dequant when available."""
    w_deq: Optional[torch.Tensor] = None
    if kernel_ext is not None and hasattr(kernel_ext, "dequantize_packed_to_fp16"):
        try:
            w_deq = kernel_ext.dequantize_packed_to_fp16(w_packed, w_scale, int(in_features), _cuda_stream_handle(dY2d))
        except Exception:
            w_deq = None
    if w_deq is None:
        w_deq = ternary_weight_deq_detached(w_shadow).to(device=dY2d.device, dtype=torch.float16)
    return dY2d.to(w_deq.dtype).matmul(w_deq).to(dtype=out_dtype)



class BitLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_shadow: torch.Tensor,
        w_packed: Optional[torch.Tensor],
        w_scale: Optional[torch.Tensor],
        w_deq_half: Optional[torch.Tensor],
        kernel_ext: Any,
        allow_fallback: bool,
        backward_mode: str,
        dx_grad: str,
        dweight_precision: str,
    ):
        original_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        if x2d.shape[1] != w_shadow.shape[1]:
            raise RuntimeError(f"BitLinear shape mismatch: x={tuple(x2d.shape)}, W={tuple(w_shadow.shape)}")

        has_packed = w_packed is not None and w_scale is not None
        has_deq = w_deq_half is not None
        use_kernel = (
            kernel_ext is not None
            and x2d.is_cuda
            and w_shadow.is_cuda
            and x2d.dtype == torch.float16
            and w_shadow.dtype == torch.float32
        )
        use_packed = bool(use_kernel and has_packed)
        if isinstance(backward_mode, bool):
            mode = "hybrid" if backward_mode else "cuda"
        else:
            mode = str(backward_mode or "hybrid").lower()
        if mode not in {"hybrid", "torch", "cuda"}:
            mode = "hybrid"
        dx_grad = str(dx_grad or "torch").lower().replace("_", "-")
        if dx_grad in {"exact", "packed", "custom-exact"}:
            dx_grad = "custom"
        if dx_grad in {"quantized", "custom-quant", "quant"}:
            dx_grad = "custom-quantized"
        if dx_grad not in {"torch", "custom", "custom-quantized"}:
            dx_grad = "torch"

        ctx.original_shape = original_shape
        ctx.use_kernel = bool(use_kernel)
        ctx.use_packed = bool(use_packed)
        ctx.has_deq = bool(has_deq and use_packed)
        ctx.kernel_ext = kernel_ext
        ctx.allow_fallback = bool(allow_fallback)
        ctx.backward_mode = mode
        ctx.dx_grad = dx_grad
        ctx.dweight_precision = str(dweight_precision or "fp16").lower()

        if use_packed:
            if has_deq:
                ctx.save_for_backward(x2d, w_shadow, w_packed, w_scale, w_deq_half)  # type: ignore[arg-type]
            else:
                ctx.save_for_backward(x2d, w_shadow, w_packed, w_scale)  # type: ignore[arg-type]
            y2d = kernel_ext.bitlinear_forward_packed(x2d, w_packed, w_scale, _cuda_stream_handle(x2d))
        else:
            ctx.save_for_backward(x2d, w_shadow)
            if use_kernel:
                y2d = kernel_ext.bitlinear_forward_from_shadow(x2d, w_shadow.contiguous(), _cuda_stream_handle(x2d))
            else:
                if not allow_fallback:
                    raise RuntimeError("Custom kernel unavailable/incompatible; add --allow-torch-fallback for PyTorch STE fallback")
                xq = fake_quant_act_int8_ste(x2d)
                wq = ternary_weight_ste(w_shadow).to(dtype=xq.dtype)
                y2d = F.linear(xq, wq)
        return y2d.reshape(*original_shape[:-1], w_shadow.shape[0])

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        saved = ctx.saved_tensors
        x2d = saved[0]
        w_shadow = saved[1]
        dY2d = dY.reshape(-1, dY.shape[-1]).contiguous()
        if dY2d.dtype != torch.float16 and ctx.use_kernel:
            dY2d = dY2d.to(torch.float16)

        if ctx.use_kernel and ctx.backward_mode == "cuda":
            dX2d, dW = ctx.kernel_ext.bitlinear_backward_from_shadow(
                x2d, dY2d, w_shadow.contiguous(), _cuda_stream_handle(dY2d)
            )
        else:
            if ctx.use_kernel and ctx.use_packed:
                w_packed, w_scale = saved[2], saved[3]
                if ctx.dx_grad == "custom":
                    dX2d = ctx.kernel_ext.bitlinear_backward_input_packed(
                        dY2d, w_packed, w_scale, x2d.shape[1], _cuda_stream_handle(dY2d)
                    )
                elif ctx.dx_grad == "custom-quantized" and hasattr(ctx.kernel_ext, "bitlinear_backward_input_packed_quantized"):
                    dX2d = ctx.kernel_ext.bitlinear_backward_input_packed_quantized(
                        dY2d, w_packed, w_scale, x2d.shape[1], _cuda_stream_handle(dY2d)
                    )
                else:
                    if ctx.has_deq and len(saved) >= 5:
                        w_deq = saved[4]
                    elif hasattr(ctx.kernel_ext, "dequantize_packed_weights_half"):
                        w_deq = ctx.kernel_ext.dequantize_packed_weights_half(
                            w_packed, w_scale, x2d.shape[1], _cuda_stream_handle(dY2d)
                        )
                    else:
                        w_deq = ternary_weight_deq_detached(w_shadow).to(device=dY2d.device, dtype=torch.float16)
                    dX2d = dY2d.to(w_deq.dtype).matmul(w_deq).to(dtype=x2d.dtype)
            else:
                wq = ternary_weight_deq_detached(w_shadow)
                dX2d = dY2d.float().matmul(wq.float()).to(dtype=x2d.dtype)
            dW = _ste_dweight_matmul(dY2d, x2d, ctx.dweight_precision)
        # forward has 10 non-ctx inputs: x, w_shadow, w_packed, w_scale, w_deq_half,
        # kernel_ext, allow_fallback, backward_mode, dx_grad, dweight_precision.
        return dX2d.reshape(ctx.original_shape), dW, None, None, None, None, None, None, None, None


class TernaryLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_ext: Any,
        allow_fallback: bool,
        backward_mode: str = "hybrid",
        dx_grad: str = "torch",
        dweight_precision: str = "fp16",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.kernel_ext = kernel_ext
        self.allow_fallback = bool(allow_fallback)
        if isinstance(backward_mode, bool):
            self.backward_mode = "hybrid" if backward_mode else "cuda"
        else:
            self.backward_mode = str(backward_mode or "hybrid").lower()
            if self.backward_mode not in {"hybrid", "torch", "cuda"}:
                self.backward_mode = "hybrid"
        self.dx_grad = str(dx_grad or "torch").lower().replace("_", "-")
        if self.dx_grad in {"exact", "packed", "custom-exact"}:
            self.dx_grad = "custom"
        if self.dx_grad in {"quantized", "custom-quant", "quant"}:
            self.dx_grad = "custom-quantized"
        if self.dx_grad not in {"torch", "custom", "custom-quantized"}:
            self.dx_grad = "torch"
        self.dweight_precision = str(dweight_precision or "fp16").lower()
        if self.dweight_precision == "tf32":
            self.dweight_precision = "fp32"
        if self.dweight_precision not in {"fp32", "bf16", "fp16"}:
            self.dweight_precision = "fp16"
        self.weight_shadow = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        nn.init.normal_(self.weight_shadow, mean=0.0, std=1.0 / math.sqrt(in_features))
        self._packed_cache_version: Optional[int] = None
        self._packed_cache: Optional[torch.Tensor] = None
        self._scale_cache: Optional[torch.Tensor] = None
        self._deq_cache: Optional[torch.Tensor] = None

    def invalidate_packed_cache(self) -> None:
        self._packed_cache_version = None
        self._packed_cache = None
        self._scale_cache = None
        self._deq_cache = None

    def _get_weight_caches(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.kernel_ext is None:
            return None, None, None
        if not (x.is_cuda and self.weight_shadow.is_cuda and x.dtype == torch.float16 and self.weight_shadow.dtype == torch.float32):
            return None, None, None
        version = int(self.weight_shadow._version)
        cache_ok = (
            self._packed_cache is not None
            and self._scale_cache is not None
            and self._packed_cache_version == version
            and self._packed_cache.device == self.weight_shadow.device
        )
        if not cache_ok:
            with torch.no_grad():
                packed, scale = self.kernel_ext.pack_ternary_weights(self.weight_shadow.contiguous(), _cuda_stream_handle(self.weight_shadow))
            self._packed_cache = packed
            self._scale_cache = scale
            self._deq_cache = None
            self._packed_cache_version = version
        if self.dx_grad == "torch" and self._deq_cache is None and self._packed_cache is not None and hasattr(self.kernel_ext, "dequantize_packed_weights_half"):
            with torch.no_grad():
                self._deq_cache = self.kernel_ext.dequantize_packed_weights_half(
                    self._packed_cache, self._scale_cache, self.in_features, _cuda_stream_handle(self.weight_shadow)
                )
        return self._packed_cache, self._scale_cache, self._deq_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_packed, w_scale, w_deq = self._get_weight_caches(x)
        return BitLinearFunction.apply(
            x,
            self.weight_shadow,
            w_packed,
            w_scale,
            w_deq,
            self.kernel_ext,
            self.allow_fallback,
            self.backward_mode,
            self.dx_grad,
            self.dweight_precision,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len].to(device=device, dtype=dtype), self.sin_cached[:seq_len].to(device=device, dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos_full = torch.repeat_interleave(cos, 2, dim=-1)[None, None, :, :]
    sin_full = torch.repeat_interleave(sin, 2, dim=-1)[None, None, :, :]
    return x * cos_full + rotate_half(x) * sin_full


_FLEX_ATTENTION_CACHE: Dict[Tuple[str, bool], Any] = {}
_FLEX_BLOCK_MASK_CACHE: Dict[Tuple[str, int, int, int, int, int], Any] = {}
_FLEX_ATTENTION_WARNED: set = set()


def _causal_mask_mod(b_idx, h_idx, q_idx, kv_idx):
    return q_idx >= kv_idx


def _get_flex_attention_callable(backend: str, compile_enabled: bool):
    key = (backend, bool(compile_enabled))
    if key in _FLEX_ATTENTION_CACHE:
        return _FLEX_ATTENTION_CACHE[key]
    from torch.nn.attention.flex_attention import flex_attention  # type: ignore
    opts = {"BLOCKS_ARE_CONTIGUOUS": True}
    if backend == "flex-fa4":
        opts["BACKEND"] = "FLASH"
    fn = partial(flex_attention, kernel_options=opts)
    if compile_enabled:
        fn = torch.compile(fn, dynamic=False, fullgraph=False)
    _FLEX_ATTENTION_CACHE[key] = fn
    return fn


def _get_causal_block_mask(q: torch.Tensor, block_size: int):
    from torch.nn.attention.flex_attention import create_block_mask  # type: ignore
    B, H, T = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    dev_index = -1 if q.device.index is None else int(q.device.index)
    key = (q.device.type, dev_index, B, H, T, int(block_size))
    if key not in _FLEX_BLOCK_MASK_CACHE:
        _FLEX_BLOCK_MASK_CACHE[key] = create_block_mask(
            _causal_mask_mod,
            B, H, T, T,
            device=q.device,
            BLOCK_SIZE=int(block_size),
        )
    return _FLEX_BLOCK_MASK_CACHE[key]


def _effective_flex_block_size(backend: str, block_size: int) -> int:
    bs = max(1, int(block_size or 256))
    if str(backend or "").lower() == "flex-fa4":
        # The FA4 FLASH backend on Hopper/Blackwell requires block-sparse masks
        # with a block size that is a multiple of 256. A user-provided 128 would
        # otherwise fail on the first forward and fall back to SDPA.
        if bs < 256:
            bs = 256
        if bs % 256 != 0:
            bs = make_divisible(bs, 256)
    return bs


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str,
    compile_enabled: bool,
    dropout_p: float,
    training: bool,
    block_size: int = 256,
) -> torch.Tensor:
    backend = str(backend or "sdpa").lower()
    if backend in {"flex", "flex-fa4"}:
        try:
            if dropout_p not in (0.0, 0) and training:
                raise RuntimeError("FlexAttention dropout path disabled; use SDPA or set attention dropout to 0")
            eff_block_size = _effective_flex_block_size(backend, block_size)
            if eff_block_size != int(block_size):
                warn_key = (backend, "block_size_adjust", str(block_size), str(eff_block_size))
                if warn_key not in _FLEX_ATTENTION_WARNED and rank0():
                    _FLEX_ATTENTION_WARNED.add(warn_key)
                    print(f"[attention] {backend} adjusted flex block size {block_size} -> {eff_block_size}.", flush=True)
            fn = _get_flex_attention_callable(backend, compile_enabled)
            block_mask = _get_causal_block_mask(q, eff_block_size)
            return fn(q, k, v, block_mask=block_mask)
        except Exception as exc:
            warn_key = (backend, type(exc).__name__, str(exc)[:160])
            if warn_key not in _FLEX_ATTENTION_WARNED and rank0():
                _FLEX_ATTENTION_WARNED.add(warn_key)
                print(f"[attention] {backend} unavailable or failed once ({exc}); falling back to SDPA.", flush=True)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p if training else 0.0, is_causal=True)

class TernaryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, kernel_ext: Any):
        super().__init__()
        assert cfg.hidden_size is not None and cfg.heads is not None
        self.hidden_size = cfg.hidden_size
        self.heads = cfg.heads
        self.head_dim = cfg.hidden_size // cfg.heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim")
        self.qkv = TernaryLinear(cfg.hidden_size, 3 * cfg.hidden_size, kernel_ext, cfg.allow_torch_fallback, cfg.use_torch_dweight_grad, cfg.bitlinear_dx_grad, cfg.bitlinear_dweight_dtype)
        self.o_proj = TernaryLinear(cfg.hidden_size, cfg.hidden_size, kernel_ext, cfg.allow_torch_fallback, cfg.use_torch_dweight_grad, cfg.bitlinear_dx_grad, cfg.bitlinear_dweight_dtype)
        self.rope = RotaryEmbedding(self.head_dim, cfg.seq_len, cfg.rope_theta)
        self.dropout = cfg.attn_dropout
        self.attention_backend = str(getattr(cfg, "attention_backend", "sdpa")).lower()
        self.flex_attention_compile = bool(getattr(cfg, "flex_attention_compile", True))
        self.flex_block_size = int(getattr(cfg, "flex_block_size", 256))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        cos, sin = self.rope(T, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = run_attention(q, k, v, self.attention_backend, self.flex_attention_compile, self.dropout, self.training, self.flex_block_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class TernaryMLP(nn.Module):
    def __init__(self, cfg: ModelConfig, kernel_ext: Any, expert: bool = False):
        super().__init__()
        assert cfg.hidden_size is not None
        hidden = cfg.intermediate_size or make_divisible(int(cfg.hidden_size * cfg.mlp_ratio), cfg.multiple_of)
        self.activation = cfg.activation
        self.hidden = hidden
        if cfg.activation == "swiglu":
            self.gate_up = TernaryLinear(cfg.hidden_size, 2 * hidden, kernel_ext, cfg.allow_torch_fallback, cfg.use_torch_dweight_grad, cfg.bitlinear_dx_grad, cfg.bitlinear_dweight_dtype)
            self.down = TernaryLinear(hidden, cfg.hidden_size, kernel_ext, cfg.allow_torch_fallback, cfg.use_torch_dweight_grad, cfg.bitlinear_dx_grad, cfg.bitlinear_dweight_dtype)
        else:
            self.up = TernaryLinear(cfg.hidden_size, hidden, kernel_ext, cfg.allow_torch_fallback, cfg.use_torch_dweight_grad, cfg.bitlinear_dx_grad, cfg.bitlinear_dweight_dtype)
            self.down = TernaryLinear(hidden, cfg.hidden_size, kernel_ext, cfg.allow_torch_fallback, cfg.use_torch_dweight_grad, cfg.bitlinear_dx_grad, cfg.bitlinear_dweight_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate, up = self.gate_up(x).split(self.hidden, dim=-1)
            return self.down(F.silu(gate) * up)
        return self.down(F.relu(self.up(x)).square())



class TernaryMoE(nn.Module):
    """Native ternary MoE with grouped dispatch and SonicMoE-style token rounding.

    This keeps the experts as TernaryMLP/BitLinear modules. It does not call SonicMoE's
    BF16 grouped-GEMM kernels directly, because those kernels expect dense expert weight
    layouts. The router/dispatch policy mirrors the Sonic idea that per-expert token counts
    should land on hardware tile multiples instead of padding arbitrary tails.
    """

    def __init__(self, cfg: ModelConfig, kernel_ext: Any):
        super().__init__()
        assert cfg.hidden_size is not None
        self.num_experts = int(cfg.num_experts)
        self.top_k = int(cfg.top_k)
        self.pad_to_multiple = int(cfg.moe_pad_to_multiple)
        self.aux_coef = float(cfg.router_aux_loss_coef)
        self.routing_backend = str(getattr(cfg, "moe_routing_backend", "grouped")).lower()
        if self.routing_backend not in {"naive", "grouped", "sonic"}:
            self.routing_backend = "grouped"
        self.topk_over_softmax = bool(getattr(cfg, "moe_topk_over_softmax", False))
        self.norm_topk_probs = bool(getattr(cfg, "moe_norm_topk_probs", True))
        self.token_rounding = bool(getattr(cfg, "moe_token_rounding", False))
        self.token_rounding_strategy = str(getattr(cfg, "moe_token_rounding_strategy", "pad")).lower()
        if self.token_rounding_strategy not in {"pad", "none", "drop", "down", "nearest", "up"}:
            self.token_rounding_strategy = "pad"
        self.router = nn.Linear(cfg.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([TernaryMLP(cfg, kernel_ext, expert=True) for _ in range(self.num_experts)])

    def _route(self, logits: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.topk_over_softmax:
            probs = F.softmax(logits.float(), dim=-1)
            top_weights, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
            if self.norm_topk_probs:
                top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            return top_idx, top_weights.to(dtype=dtype), probs
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_vals.float(), dim=-1).to(dtype=dtype)
        probs = F.softmax(logits.float(), dim=-1)
        return top_idx, top_weights, probs

    def _target_count(self, real: int) -> int:
        tile = max(1, int(self.pad_to_multiple))
        if real <= 0 or not self.token_rounding or self.routing_backend != "sonic" or tile <= 1:
            return real
        strat = self.token_rounding_strategy
        if strat in {"none", "pad"}:
            return real
        if strat in {"drop", "down"}:
            return (real // tile) * tile
        if strat == "up":
            return make_divisible(real, tile)
        # nearest: bounded deviation to the nearest tile, but avoid zeroing small experts.
        return max(tile, int(round(float(real) / float(tile))) * tile)

    def _expert_call(self, expert_id: int, x_e: torch.Tensor, real: int, C: int) -> torch.Tensor:
        if real <= 0:
            return x_e.new_zeros((0, C))
        if self.pad_to_multiple > 1:
            padded = make_divisible(real, self.pad_to_multiple)
            if padded != real:
                x_e = torch.cat([x_e, torch.zeros(padded - real, C, device=x_e.device, dtype=x_e.dtype)], dim=0)
        y_e = self.experts[expert_id](x_e.reshape(1, x_e.shape[0], C)).reshape(x_e.shape[0], C)
        return y_e[:real]

    def _forward_naive(self, flat: torch.Tensor, top_idx: torch.Tensor, top_weights: torch.Tensor, C: int) -> torch.Tensor:
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            pos, slot = torch.where(top_idx == expert_id)
            if pos.numel() == 0:
                continue
            real = int(pos.numel())
            x_e = flat.index_select(0, pos)
            y_e = self._expert_call(expert_id, x_e, real, C)
            out.index_add_(0, pos, y_e * top_weights[pos, slot].unsqueeze(-1))
        return out

    def _round_assignments(
        self,
        flat: torch.Tensor,
        expert_id: int,
        pos: torch.Tensor,
        w: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        real = int(pos.numel())
        target = self._target_count(real)
        if target == real or self.routing_backend != "sonic" or not self.token_rounding:
            return pos, w
        if target <= 0:
            return pos[:0], w[:0]
        if target < real:
            # Drop the least-confident tail assignments, not just the last sorted entries.
            keep = torch.topk(w.float(), k=target, largest=True, sorted=False).indices
            return pos.index_select(0, keep), w.index_select(0, keep)

        # Target is above the top-k token-choice count. Approximate Sonic token rounding by
        # expert-choice fill: add highest-probability tokens that were not already routed here.
        extra = target - real
        num_tok = flat.shape[0]
        if extra <= 0 or num_tok <= real:
            return pos, w
        assigned = torch.zeros(num_tok, device=flat.device, dtype=torch.bool)
        assigned.scatter_(0, pos, True)
        scores = probs[:, expert_id].to(torch.float32).masked_fill(assigned, float("-inf"))
        available = int((~assigned).sum().item())
        if available <= 0:
            return pos, w
        extra = min(extra, available)
        extra_vals, extra_pos = torch.topk(scores, k=extra, largest=True, sorted=False)
        valid = torch.isfinite(extra_vals)
        if valid.any():
            extra_pos = extra_pos[valid]
            extra_w = extra_vals[valid].to(dtype=w.dtype)
            return torch.cat([pos, extra_pos], dim=0), torch.cat([w, extra_w], dim=0)
        return pos, w

    def _forward_grouped(self, flat: torch.Tensor, top_idx: torch.Tensor, top_weights: torch.Tensor, probs: torch.Tensor, C: int) -> torch.Tensor:
        num_tok = flat.shape[0]
        tok = torch.arange(num_tok, device=flat.device, dtype=torch.long).repeat_interleave(self.top_k)
        expert = top_idx.reshape(-1).contiguous()
        weight = top_weights.reshape(-1).contiguous()
        order = torch.argsort(expert, stable=False)
        expert = expert.index_select(0, order)
        tok = tok.index_select(0, order)
        weight = weight.index_select(0, order)
        counts = torch.bincount(expert, minlength=self.num_experts)
        offsets = torch.empty(self.num_experts + 1, device=flat.device, dtype=torch.long)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(counts, dim=0)
        out = torch.zeros_like(flat)
        for expert_id in range(self.num_experts):
            start = int(offsets[expert_id].item())
            end = int(offsets[expert_id + 1].item())
            if end <= start:
                continue
            pos = tok[start:end]
            w = weight[start:end]
            if self.routing_backend == "sonic" and self.token_rounding:
                pos, w = self._round_assignments(flat, expert_id, pos, w, probs)
                if pos.numel() == 0:
                    continue
            real = int(pos.numel())
            x_e = flat.index_select(0, pos)
            y_e = self._expert_call(expert_id, x_e, real, C)
            out.index_add_(0, pos, y_e * w.unsqueeze(-1))
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        flat = x.reshape(B * T, C)
        logits = self.router(flat.float())
        top_idx, top_weights, probs = self._route(logits, x.dtype)
        if self.routing_backend == "naive":
            out = self._forward_naive(flat, top_idx, top_weights, C)
        else:
            out = self._forward_grouped(flat, top_idx, top_weights, probs, C)
        top1 = top_idx[:, 0]
        density = F.one_hot(top1, num_classes=self.num_experts).float().mean(dim=0)
        density_proxy = probs.mean(dim=0)
        aux = self.num_experts * torch.sum(density * density_proxy) * self.aux_coef
        return out.reshape(B, T, C), aux


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, kernel_ext: Any, layer_idx: int, moe_layers: Sequence[int]):
        super().__init__()
        assert cfg.hidden_size is not None
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = TernaryAttention(cfg, kernel_ext)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.is_moe = cfg.architecture == "moe" and layer_idx in set(moe_layers)
        self.ffn = TernaryMoE(cfg, kernel_ext) if self.is_moe else TernaryMLP(cfg, kernel_ext)
        self.dropout = nn.Dropout(cfg.resid_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.attn(self.norm1(x)))
        if self.is_moe:
            y, aux = self.ffn(self.norm2(x))  # type: ignore[misc]
        else:
            y = self.ffn(self.norm2(x))  # type: ignore[operator]
            aux = x.new_zeros((), dtype=torch.float32)
        x = x + self.dropout(y)
        return x, aux.float()


def resolve_moe_layers(cfg: ModelConfig) -> List[int]:
    if cfg.architecture != "moe" or cfg.moe_num_layers <= 0:
        return []
    ids: List[int] = []
    i = cfg.layers - 1
    while i >= 0 and len(ids) < cfg.moe_num_layers:
        ids.append(i)
        i -= max(1, cfg.moe_layer_stride)
    return sorted(set(ids))


class TernaryTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig, kernel_ext: Any):
        super().__init__()
        assert cfg.hidden_size is not None
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.resid_dropout)
        moe_layers = resolve_moe_layers(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, kernel_ext, i, moe_layers) for i in range(cfg.layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.tie_embeddings = bool(cfg.tie_embeddings)
        if cfg.ternarize_lm_head:
            self.lm_head = TernaryLinear(cfg.hidden_size, cfg.vocab_size, kernel_ext, cfg.allow_torch_fallback, cfg.use_torch_dweight_grad, cfg.bitlinear_dx_grad, cfg.bitlinear_dweight_dtype)
        elif not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None
        self.apply(self._init_non_ternary)

    def _init_non_ternary(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        runtime_dtype_name = getattr(self.cfg, "runtime_dtype", "fp16")
        runtime_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(runtime_dtype_name, torch.float16)
        if self.cfg.use_custom_kernel:
            runtime_dtype = torch.float16  # EL CUDA BitLinear currently expects FP16 activations.
        x = self.tok_embeddings(input_ids).to(runtime_dtype)
        x = self.drop(x)
        aux = x.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x, block_aux = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x, block_aux = block(x)
            aux = aux + block_aux
        x = self.norm(x)
        return x, aux

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            return F.linear(hidden.float(), self.tok_embeddings.weight.float())
        elif isinstance(self.lm_head, TernaryLinear):
            return self.lm_head(hidden).float()
        # Non-ternary LM head parameters are kept FP32 for stable AMP/scaler behavior.
        return F.linear(hidden.float(), self.lm_head.weight.float()).float()

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, aux = self.forward_hidden(input_ids)
        if return_hidden:
            return hidden, aux
        return self.project_logits(hidden), aux

# =============================================================================
# Shape inference and parameter counts
# =============================================================================


def choose_heads(hidden: int) -> int:
    candidates = [h for h in range(1, min(hidden, 128) + 1) if hidden % h == 0]
    return min(candidates, key=lambda h: abs(hidden // h - 128)) if candidates else 1


def estimate_params_for_hidden(cfg: ModelConfig, hidden: int) -> int:
    inter = cfg.intermediate_size or make_divisible(int(cfg.mlp_ratio * hidden), cfg.multiple_of)
    total = cfg.vocab_size * hidden
    if not cfg.tie_embeddings:
        total += cfg.vocab_size * hidden
    total += hidden
    tmp = dataclasses.replace(cfg, hidden_size=hidden)
    moe_ids = set(resolve_moe_layers(tmp))
    for i in range(cfg.layers):
        total += 4 * hidden * hidden  # qkv + o
        total += 2 * hidden           # norms
        if cfg.architecture == "moe" and i in moe_ids:
            total += hidden * cfg.num_experts
            expert_mlp = (3 * hidden * inter) if cfg.activation == "swiglu" else (2 * hidden * inter)
            total += cfg.num_experts * expert_mlp
        else:
            total += (3 * hidden * inter) if cfg.activation == "swiglu" else (2 * hidden * inter)
    return int(total)


def resolve_model_config(cfg: ModelConfig) -> ModelConfig:
    if cfg.hidden_size is None:
        if cfg.target_params is None:
            raise ValueError("Set either --hidden-size or --target-params")
        heads = cfg.heads or 8
        multiple = max(cfg.multiple_of, heads * 2)
        lo = make_divisible(max(64, heads * 32), multiple)
        hi = lo
        target = int(cfg.target_params)
        while estimate_params_for_hidden(dataclasses.replace(cfg, heads=heads), hi) < target:
            hi *= 2
        best = lo
        while lo <= hi:
            mid = make_divisible((lo + hi) // 2, multiple)
            est = estimate_params_for_hidden(dataclasses.replace(cfg, heads=heads), mid)
            if est <= target:
                best = mid
                lo = mid + multiple
            else:
                hi = mid - multiple
        cfg = dataclasses.replace(cfg, hidden_size=best, heads=heads)
    if cfg.heads is None:
        cfg = dataclasses.replace(cfg, heads=choose_heads(cfg.hidden_size))
    if cfg.hidden_size % cfg.heads != 0:
        raise ValueError(f"hidden_size={cfg.hidden_size} must be divisible by heads={cfg.heads}")
    if (cfg.hidden_size // cfg.heads) % 2 != 0:
        raise ValueError("RoPE requires even head_dim; adjust --hidden-size or --heads")
    if cfg.intermediate_size is None:
        cfg = dataclasses.replace(cfg, intermediate_size=make_divisible(int(cfg.hidden_size * cfg.mlp_ratio), cfg.multiple_of))
    return cfg


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    ternary_shadow = sum(p.numel() for n, p in model.named_parameters() if n.endswith("weight_shadow"))
    return total, ternary_shadow


def cast_non_shadow_params(model: nn.Module, dtype: torch.dtype) -> None:
    """Legacy helper kept for experiments; not used by the default trainer.

    Native AMP training should keep trainable parameters in FP32 and cast only
    runtime activations. Casting optimizer-owned parameters to FP16 causes
    GradScaler.unscale_() to reject FP16 gradients.
    """
    for name, p in model.named_parameters():
        if name.endswith("weight_shadow"):
            p.data = p.data.float()
        else:
            p.data = p.data.to(dtype=dtype)


def keep_trainable_params_fp32(model: nn.Module) -> None:
    """Keep all optimizer-owned trainable floating parameters in FP32.

    The custom BitLinear path still runs FP16 activations into FP32 shadow
    weights. Keeping embeddings, router weights, norm weights, LM-head weights,
    and all BitLinear shadow weights as FP32 gives GradScaler FP32 gradients to
    unscale and lets AdamW/bitsandbytes maintain a stable master state.
    """
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.float()




def invalidate_ternary_packed_caches(model: nn.Module) -> None:
    """Drop cached packed ternary weights after optimizer updates.

    Caches are keyed by Parameter._version, but explicitly clearing after optimizer.step()
    makes this robust with optimizers that mutate parameters through fused/foreach paths.
    """
    for module in model.modules():
        invalidate = getattr(module, "invalidate_packed_cache", None)
        if callable(invalidate):
            invalidate()

def assert_no_fp16_trainable_params(model: nn.Module) -> None:
    offenders = [(name, tuple(p.shape)) for name, p in model.named_parameters() if p.requires_grad and p.dtype == torch.float16]
    if offenders:
        preview = ", ".join(f"{name}{shape}" for name, shape in offenders[:8])
        more = "" if len(offenders) <= 8 else f" ... and {len(offenders) - 8} more"
        raise RuntimeError(
            "FP16 trainable parameters detected. Keep trainable parameters/shadow weights FP32 and use "
            "FP16/BF16 only for runtime activations/autocast. Offenders: " + preview + more
        )


def parameter_dtype_summary(model: nn.Module) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        key = str(p.dtype).replace("torch.", "")
        counts[key] = counts.get(key, 0) + p.numel()
    return counts


def format_dtype_summary(summary: Dict[str, int]) -> str:
    if not summary:
        return "<no trainable parameters>"
    return ", ".join(f"{dtype}={human_int(count)}" for dtype, count in sorted(summary.items()))


def make_grad_scaler(dtype_name: str, device: torch.device):
    enabled = bool(device.type == "cuda" and dtype_name == "fp16")
    # torch.amp.GradScaler('cuda', ...) is the newer API; keep a fallback for
    # older PyTorch wheels.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def scaler_is_enabled(scaler: Any) -> bool:
    try:
        return bool(scaler.is_enabled())
    except Exception:
        return False


# =============================================================================
# Optimizer, schedule, save/export
# =============================================================================


def param_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "norm" in lname or "embedding" in lname:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    groups = param_groups(model, cfg.weight_decay_peak)
    if cfg.optimizer == "adam8bit":
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError("--optimizer adam8bit requires bitsandbytes") from exc
        return bnb.optim.Adam8bit(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps)
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps, fused=True)


def lr_wd_for_step(step: int, cfg: TrainConfig) -> Tuple[float, float]:
    warmup = max(1, cfg.warmup_steps)
    cooldown_start = max(warmup + 1, int(cfg.max_steps * cfg.cooldown_start_frac))
    cooldown_start = min(cooldown_start, max(1, cfg.max_steps - 1))
    cooldown_lr = cfg.cooldown_lr if cfg.cooldown_lr is not None else max(cfg.min_lr, cfg.learning_rate * 0.1)
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup, cfg.weight_decay_peak * (step + 1) / warmup
    if step < cooldown_start:
        t = (step - warmup) / max(1, cooldown_start - warmup)
        lr = cooldown_lr + 0.5 * (cfg.learning_rate - cooldown_lr) * (1 + math.cos(math.pi * t))
        return lr, cfg.weight_decay_peak
    t = min(1.0, (step - cooldown_start) / max(1, cfg.max_steps - cooldown_start))
    lr = cfg.min_lr + 0.5 * (cooldown_lr - cfg.min_lr) * (1 + math.cos(math.pi * t))
    return lr, 0.0


def apply_lr_wd(opt: torch.optim.Optimizer, lr: float, wd: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr
        # Keep no-decay groups at zero.
        if group.get("weight_decay", 0.0) != 0.0:
            group["weight_decay"] = wd


def pack_ternary_cpu(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w_cpu = w.detach().float().cpu().contiguous()
    K, N = w_cpu.shape
    scale = w_cpu.abs().mean(dim=1).clamp_min(1e-8).contiguous()
    tern = torch.round(w_cpu / scale[:, None]).clamp(-1, 1).to(torch.int8).numpy()
    words = (N + 15) // 16
    padded = np.zeros((K, words * 16), dtype=np.int8)
    padded[:, :N] = tern
    t = padded.reshape(K, words, 16)
    code = ((t > 0).astype(np.uint32) | ((t < 0).astype(np.uint32) << 1))
    packed = np.zeros((K, words), dtype=np.uint32)
    for lane in range(16):
        packed |= code[:, :, lane] << np.uint32(2 * lane)
    return torch.from_numpy(packed.view(np.int32)).contiguous(), scale.contiguous()


def collect_shadow_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    base = unwrap_model(model)
    return {k: v.detach().cpu().contiguous() for k, v in base.state_dict().items()}


def collect_packed_ternary_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    base = unwrap_model(model)
    out: Dict[str, torch.Tensor] = {}
    ternary_weight_names = set()
    for name, module in base.named_modules():
        if isinstance(module, TernaryLinear):
            packed, scale = pack_ternary_cpu(module.weight_shadow)
            prefix = f"{name}." if name else ""
            out[prefix + "weight_packed_i32"] = packed
            out[prefix + "weight_scale_fp32"] = scale
            ternary_weight_names.add(prefix + "weight_shadow")
    for name, tensor in base.state_dict().items():
        if name in ternary_weight_names:
            continue
        out[name] = tensor.detach().cpu().contiguous()
    return out


def _write_checkpoint_files(model: nn.Module, model_cfg: ModelConfig, train_cfg: TrainConfig, step: int) -> None:
    if not rank0():
        return
    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_config.json").write_text(json.dumps(asdict(model_cfg), indent=2))
    (out_dir / "train_config.json").write_text(json.dumps(asdict(train_cfg), indent=2))
    shadow_path = out_dir / f"checkpoint_step_{step:08d}.shadow.safetensors"
    tern_path = out_dir / f"checkpoint_step_{step:08d}.ternary.safetensors"
    safe_save_file(collect_shadow_state(model), str(shadow_path), metadata={"format": "fp32_shadow", "step": str(step)})
    safe_save_file(
        collect_packed_ternary_state(model),
        str(tern_path),
        metadata={
            "format": "packed_ternary_w1.58a8",
            "ternary_scheme": "16 two-bit weights per int32; 00=0, 01=+1, 10=-1, 11=reserved_zero; per-row absmean scale",
            "activation_scheme": "INT8 per-token dynamic scale in CUDA kernel",
            "step": str(step),
        },
    )
    log(f"Saved:\n  {shadow_path}\n  {tern_path}")


def save_checkpoint(model: nn.Module, model_cfg: ModelConfig, train_cfg: TrainConfig, step: int) -> None:
    if safe_save_file is None:
        raise RuntimeError("safetensors is required: pip install safetensors")
    # FSDP needs all ranks to enter summon_full_params; only rank 0 writes files.
    if train_cfg.fsdp and torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(model, writeback=False, recurse=True):
            _write_checkpoint_files(model, model_cfg, train_cfg, step)
    else:
        _write_checkpoint_files(model, model_cfg, train_cfg, step)


# =============================================================================
# Train
# =============================================================================


def enable_flash_sdp() -> None:
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def _parse_bool_auto(value: str, default: bool) -> bool:
    v = str(value or "auto").lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    return bool(default)


def effective_ddp_flags(cfg: TrainConfig, model_cfg: ModelConfig, world: int) -> Tuple[bool, bool, bool]:
    """Return (find_unused, static_graph, gradient_as_bucket_view) for the actual DDP wrapper."""
    auto_unused = model_cfg.architecture == "moe"
    find_unused = _parse_bool_auto(cfg.ddp_find_unused_parameters, auto_unused)
    static_graph = bool(cfg.ddp_static_graph) and not find_unused
    if world > 1 and static_graph and bool(cfg.ddp_no_sync) and int(cfg.grad_accum_steps) > 1:
        log("DDP compatibility: --ddp-static-graph was requested but disabled because gradient accumulation uses DDP.no_sync; this avoids PyTorch reducer expect_autograd_hooks_ asserts.")
        static_graph = False
    if world > 1 and static_graph and (bool(model_cfg.use_custom_kernel) or bool(cfg.use_liger_fused_ce) or str(model_cfg.attention_backend).startswith("flex")):
        log("DDP compatibility: --ddp-static-graph was requested but disabled for the custom-autograd / fused-CE / FlexAttention path. Re-enable only after a baseline DDP run succeeds.")
        static_graph = False
    setattr(cfg, "_effective_ddp_find_unused", find_unused)
    setattr(cfg, "_effective_ddp_static_graph", static_graph)
    setattr(cfg, "_effective_ddp_gradient_as_bucket_view", bool(cfg.ddp_gradient_as_bucket_view))
    return find_unused, static_graph, bool(cfg.ddp_gradient_as_bucket_view)


def maybe_wrap_distributed(model: nn.Module, cfg: TrainConfig, model_cfg: ModelConfig, world: int, local_rank: int) -> nn.Module:
    if world <= 1:
        return model
    if cfg.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=local_rank)
    find_unused, static_graph, grad_bucket_view = effective_ddp_flags(cfg, model_cfg, world)
    ddp_kwargs = dict(
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=grad_bucket_view,
    )
    if static_graph:
        ddp_kwargs["static_graph"] = True
    return torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)


def maybe_no_sync(model: nn.Module, train_cfg: TrainConfig, micro_idx: int) -> contextlib.AbstractContextManager:
    if (
        train_cfg.distributed
        and train_cfg.ddp_no_sync
        and not train_cfg.fsdp
        and train_cfg.grad_accum_steps > 1
        and micro_idx < train_cfg.grad_accum_steps - 1
        and hasattr(model, "no_sync")
    ):
        return model.no_sync()  # type: ignore[return-value]
    return contextlib.nullcontext()




def lm_head_weight_for_fused_ce(model: nn.Module) -> Optional[torch.Tensor]:
    base = unwrap_model(model)
    lm_head = getattr(base, "lm_head", None)
    if lm_head is None:
        emb = getattr(base, "tok_embeddings", None)
        return getattr(emb, "weight", None)
    if isinstance(lm_head, nn.Linear):
        return lm_head.weight
    return None


def can_use_liger_fused_ce(model: nn.Module) -> bool:
    return lm_head_weight_for_fused_ce(model) is not None

def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    rank, world, local_rank, device = init_distributed()
    train_cfg.distributed = world > 1
    torch.manual_seed(train_cfg.seed + rank)
    np.random.seed(train_cfg.seed + rank)
    random.seed(train_cfg.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if train_cfg.use_flash_sdpa:
        enable_flash_sdp()

    # Data setup. In --stream-train mode, read HF rows directly without materializing a token cache.
    tokens_path: Optional[Path] = None
    if train_cfg.stream_train:
        if not train_cfg.hf_dataset:
            raise ValueError("--stream-train requires --hf-dataset")
        tokenizer = load_training_tokenizer(train_cfg)
        meta = {
            "tokenizer": train_cfg.tokenizer,
            "auto_tokenizer_fallback": train_cfg.auto_tokenizer_fallback,
            "vocab_size": int(train_cfg.vocab_size_override or len(tokenizer)),
            "num_tokens": None,
            "stream_train": True,
            "hf_dataset": list(train_cfg.hf_dataset),
        }
        log("HF direct streaming enabled; no token cache will be built.")
    else:
        # Only rank 0 builds token cache; others wait and reuse.
        if world > 1 and rank != 0:
            _distributed_barrier_if_ready()
        if rank == 0:
            tokens_path, meta, tokenizer = build_or_load_token_cache(train_cfg)
        if world > 1 and rank == 0:
            _distributed_barrier_if_ready()
        if rank != 0:
            read_cfg = dataclasses.replace(train_cfg, rebuild_token_cache=False)
            tokens_path, meta, tokenizer = build_or_load_token_cache(read_cfg)

    model_cfg = dataclasses.replace(model_cfg, vocab_size=int(meta["vocab_size"]))
    model_cfg = resolve_model_config(model_cfg)
    model_cfg = dataclasses.replace(
        model_cfg,
        use_custom_kernel=train_cfg.use_custom_kernel,
        allow_torch_fallback=train_cfg.allow_torch_fallback,
        use_torch_dweight_grad=train_cfg.use_torch_dweight_grad,
        bitlinear_dx_grad=train_cfg.bitlinear_dx_grad,
        bitlinear_dweight_dtype=train_cfg.bitlinear_dweight_dtype,
        attention_backend=train_cfg.attention_backend,
        flex_attention_compile=train_cfg.flex_attention_compile,
        flex_block_size=_effective_flex_block_size(train_cfg.attention_backend, train_cfg.flex_block_size),
    )
    if int(model_cfg.flex_block_size) != int(train_cfg.flex_block_size):
        log(f"Runtime config: flex block size {train_cfg.flex_block_size} -> {model_cfg.flex_block_size} for {train_cfg.attention_backend}.")
        train_cfg = dataclasses.replace(train_cfg, flex_block_size=int(model_cfg.flex_block_size))
    setattr(model_cfg, "runtime_dtype", train_cfg.dtype)

    kernel_ext = load_custom_cuda_extension(train_cfg)
    model = TernaryTransformerLM(model_cfg, kernel_ext).to(device)
    train_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[train_cfg.dtype]
    # Keep optimizer-owned trainable parameters in FP32. The model forward still
    # casts activations to train_cfg.dtype, and the custom CUDA BitLinear kernel
    # currently requires FP16 activations with FP32 shadow weights. This avoids
    # PyTorch GradScaler's "Attempting to unscale FP16 gradients" failure.
    keep_trainable_params_fp32(model)

    total, ternary_shadow = count_parameters(model)
    log(f"Resolved model config: {json.dumps(asdict(model_cfg), indent=2)}")
    log(f"Parameters: total/shadow={human_int(total)} ternary_shadow_core={human_int(ternary_shadow)}")
    log(f"Trainable parameter dtypes: {format_dtype_summary(parameter_dtype_summary(model))}")
    if train_cfg.dtype == "fp16" and train_cfg.use_custom_kernel:
        log("Runtime activations: fp16 for the custom W1.58A8 CUDA kernel; optimizer parameters remain fp32.")

    if train_cfg.compile:
        model = torch.compile(model, mode=train_cfg.compile_mode, fullgraph=False)
        log(f"torch.compile enabled: mode={train_cfg.compile_mode}")

    model = maybe_wrap_distributed(model, train_cfg, model_cfg, world, local_rank)
    assert_no_fp16_trainable_params(model)
    optimizer = make_optimizer(train_cfg, model)
    scaler = make_grad_scaler(train_cfg.dtype, device)
    log(f"AMP GradScaler enabled: {scaler_is_enabled(scaler)}")
    if model_cfg.use_custom_kernel:
        if model_cfg.use_torch_dweight_grad:
            log(f"BitLinear backward path: forward=cached packed W1.58A8, dX={model_cfg.bitlinear_dx_grad}, dW=torch.matmul/{model_cfg.bitlinear_dweight_dtype}")
        else:
            log("BitLinear backward path: full extension scalar dW debug path")
    log(f"Attention backend: {model_cfg.attention_backend}{' (compiled)' if model_cfg.flex_attention_compile and model_cfg.attention_backend.startswith('flex') else ''}; flex_block_size={model_cfg.flex_block_size}")
    if model_cfg.architecture == "moe":
        log(f"MoE routing backend: {model_cfg.moe_routing_backend}; topk_over_softmax={model_cfg.moe_topk_over_softmax}; token_rounding={model_cfg.moe_token_rounding}")
    if train_cfg.distributed:
        log(
            f"Distributed: world={world} fsdp={train_cfg.fsdp} ddp_no_sync={train_cfg.ddp_no_sync} "
            f"ddp_find_unused={getattr(train_cfg, '_effective_ddp_find_unused', train_cfg.ddp_find_unused_parameters)} "
            f"ddp_static_graph_requested={train_cfg.ddp_static_graph} "
            f"ddp_static_graph_effective={getattr(train_cfg, '_effective_ddp_static_graph', False)} "
            f"gradient_as_bucket_view={getattr(train_cfg, '_effective_ddp_gradient_as_bucket_view', train_cfg.ddp_gradient_as_bucket_view)}"
        )
    if train_cfg.stream_train:
        if train_cfg.distributed and rank0():
            log(f"HF streaming DDP strategy: {train_cfg.hf_ddp_shard_strategy} (use 'stride' for single-source Hub streams; force 'split' only for many-shard datasets)")
        data = StreamingTokenBatcher(train_cfg, tokenizer, model_cfg.seq_len, rank=rank, world=world)
    else:
        assert tokens_path is not None
        data = TokenMemmap(tokens_path, model_cfg.seq_len)
    tokens_per_step = model_cfg.seq_len * train_cfg.batch_size * train_cfg.grad_accum_steps * world

    liger_ce = None
    liger_fused_ce = None
    if train_cfg.use_liger:
        if train_cfg.use_liger_fused_ce and can_use_liger_fused_ce(model):
            try:
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # type: ignore
                liger_fused_ce = LigerFusedLinearCrossEntropyLoss()
                log("LigerFusedLinearCrossEntropyLoss enabled; skipping explicit logits materialization.")
            except Exception as exc:
                log(f"Liger fused CE unavailable ({exc}); trying standard LigerCrossEntropyLoss.")
        if liger_fused_ce is None:
            try:
                from liger_kernel.transformers import LigerCrossEntropyLoss  # type: ignore
                liger_ce = LigerCrossEntropyLoss()
                log("LigerCrossEntropyLoss enabled.")
            except Exception as exc:
                log(f"Liger unavailable ({exc}); using torch.nn.functional.cross_entropy.")

    model.train()
    started = time.time()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step in range(train_cfg.max_steps):
        lr, wd = lr_wd_for_step(step, train_cfg)
        apply_lr_wd(optimizer, lr, wd)
        step_loss = 0.0
        for micro_idx in range(train_cfg.grad_accum_steps):
            with maybe_no_sync(model, train_cfg, micro_idx):
                x, y = data.sample_batch(train_cfg.batch_size, device)
                flat_labels = y.reshape(-1)
                if liger_fused_ce is not None:
                    hidden, aux = model(x, return_hidden=True)
                    head_weight = lm_head_weight_for_fused_ce(model)
                    if head_weight is None:
                        raise RuntimeError("Liger fused CE selected but tied/dense LM head weight is unavailable")
                    try:
                        hidden_flat = hidden.reshape(-1, hidden.size(-1)).to(dtype=head_weight.dtype)
                        lm_loss = liger_fused_ce(head_weight, hidden_flat, flat_labels)
                    except Exception as exc:
                        # Some liger-kernel versions expose the fused class but support only a subset
                        # of dtypes/layouts. Fall back without aborting the run.
                        log(f"Liger fused CE failed once ({exc}); falling back to explicit logits + CE.")
                        liger_fused_ce = None
                        logits = unwrap_model(model).project_logits(hidden)
                        flat_logits = logits.view(-1, logits.size(-1))
                        lm_loss = liger_ce(flat_logits, flat_labels) if liger_ce is not None else F.cross_entropy(flat_logits, flat_labels)
                else:
                    logits, aux = model(x)
                    flat_logits = logits.view(-1, logits.size(-1))
                    lm_loss = liger_ce(flat_logits, flat_labels) if liger_ce is not None else F.cross_entropy(flat_logits, flat_labels)
                loss = (lm_loss + aux) / train_cfg.grad_accum_steps
                if scaler_is_enabled(scaler):
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step_loss += float((lm_loss + aux).detach().cpu())
        if train_cfg.grad_clip > 0:
            if scaler_is_enabled(scaler):
                try:
                    scaler.unscale_(optimizer)
                except ValueError as exc:
                    if "Attempting to unscale FP16 gradients" in str(exc):
                        assert_no_fp16_trainable_params(model)
                    raise
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        if scaler_is_enabled(scaler):
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        invalidate_ternary_packed_caches(model)
        optimizer.zero_grad(set_to_none=True)
        running_loss += step_loss / train_cfg.grad_accum_steps

        if rank0() and (step + 1) % train_cfg.log_interval == 0:
            elapsed = max(1e-6, time.time() - started)
            done = step + 1
            avg = running_loss / train_cfg.log_interval
            log(f"step={done:,}/{train_cfg.max_steps:,} loss={avg:.4f} lr={lr:.3e} wd={wd:.3e} tok/s={done * tokens_per_step / elapsed:,.0f}")
            running_loss = 0.0

        if train_cfg.save_interval > 0 and (step + 1) % train_cfg.save_interval == 0:
            save_checkpoint(model, model_cfg, train_cfg, step + 1)
        if train_cfg.max_tokens is not None and (step + 1) * tokens_per_step >= train_cfg.max_tokens:
            break

    if world > 1:
        _distributed_barrier_if_ready()
    save_checkpoint(model, model_cfg, train_cfg, min(train_cfg.max_steps, step + 1))
    if world > 1:
        _distributed_barrier_if_ready()
    cleanup_distributed()


# =============================================================================
# CLI
# =============================================================================


def _normalize_cli_argv(argv: List[str]) -> List[str]:
    """Normalize common flag spelling variants while keeping argparse strict."""
    aliases = {
        "extension-verbose": "--extension-verbose",
        "extension_verbose": "--extension-verbose",
        "--extension_verbose": "--extension-verbose",
        "--ext-verbose": "--extension-verbose",
        "--verbose-extension": "--extension-verbose",
        "--cuda-extension-verbose": "--extension-verbose",
        "--kernel-verbose": "--extension-verbose",
        "--hf-streaming-ddp-strategy": "--hf-ddp-shard-strategy",
        "--hf-ddp-strategy": "--hf-ddp-shard-strategy",
    }
    return [aliases.get(arg, arg) for arg in argv]


def parse_args() -> Tuple[ModelConfig, TrainConfig]:
    sys.argv = _normalize_cli_argv(sys.argv)
    p = argparse.ArgumentParser(description="Pretrain dense/MoE W1.58A8 ternary LLM with EL custom CUDA BitLinear kernel")

    # Local and Hugging Face data. --data is optional when --hf-dataset or an existing --token-cache is used.
    p.add_argument("--data", nargs="*", default=[], help="Local JSON/JSONL/Markdown/Parquet paths or globs")
    p.add_argument("--hf-dataset", action="append", default=[], help="Hugging Face dataset repo or loader name. Repeat for multiple datasets.")
    p.add_argument("--hf-config", default=None, help="Optional HF dataset config/name")
    p.add_argument("--hf-split", default="train", help="HF dataset split")
    p.add_argument("--hf-streaming", action="store_true", help="Load HF data with streaming=True")
    p.add_argument("--stream-train", action="store_true", help="Train directly from streaming data instead of first building a token cache")
    p.add_argument("--hf-data-files", nargs="+", default=None, help="Optional HF/local data_files value; accepts JSON dict/list or one/more patterns")
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--hf-revision", default=None)
    p.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable containing a private HF token, if needed")
    p.add_argument("--hf-trust-remote-code", action="store_true")
    p.add_argument("--hf-shuffle-buffer", type=int, default=0, help="Approximate shuffle buffer for streaming HF datasets")
    p.add_argument("--hf-skip", type=int, default=0)
    p.add_argument("--hf-take", type=parse_count, default=None, help="Take this many HF records/examples before tokenization")
    p.add_argument("--hf-interleave-probabilities", default=None, help="Comma-separated or JSON probabilities when repeating --hf-dataset")
    p.add_argument(
        "--hf-ddp-shard-strategy",
        "--hf_ddp_shard_strategy",
        dest="hf_ddp_shard_strategy",
        choices=["auto", "stride", "split", "shard", "none"],
        default="stride",
        help=(
            "Distributed HF streaming split strategy. stride is the safe default for Hub streams; auto uses rank-stride when the stream "
            "when the stream has fewer data shards than DDP ranks; split/shard are faster only when enough source shards exist; "
            "split/shard are faster only when enough source shards exist."
        ),
    )

    # Tokenizer. Dataset IDs and tokenizer IDs are intentionally independent.
    p.add_argument("--tokenizer", default="auto", help="Tokenizer repo/path. Placeholder values use --auto-tokenizer-fallback.")
    p.add_argument("--auto-tokenizer-fallback", default="gpt2", help="Tokenizer used when --tokenizer is auto or the old example placeholder")
    p.add_argument("--tokenizer-cache-dir", default=None)
    p.add_argument("--tokenizer-revision", default=None)
    p.add_argument("--tokenizer-token-env", default=None)
    p.add_argument("--tokenizer-trust-remote-code", action="store_true")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-format", default="auto", choices=["auto", "json", "markdown", "parquet", "paraquet"])
    p.add_argument("--text-column", default="text")
    p.add_argument("--token-column", default=None, help="Optional pre-tokenized token ID column in JSON/Parquet/HF records")
    p.add_argument("--token-cache", default=None)
    p.add_argument("--rebuild-token-cache", action="store_true")
    p.add_argument("--token-cache-max-tokens", type=parse_count, default=None, help="Optional cap while building a uint32 token cache, e.g. 10B")
    p.add_argument("--vocab-size-override", type=int, default=None, help="Override tokenizer vocab size in model config")
    p.add_argument("--no-append-eos", action="store_true")
    p.add_argument("--tokenization-batch-size", type=int, default=512)
    p.add_argument("--parquet-batch-size", type=int, default=4096)

    p.add_argument("--architecture", choices=["dense", "moe"], default="dense")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--target-params", default=None, help="Approximate parameter target, e.g. 125M, 3B")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--multiple-of", type=int, default=64)
    p.add_argument("--activation", choices=["relu2", "swiglu"], default="relu2")
    p.add_argument("--no-tie-embeddings", action="store_true")
    p.add_argument("--ternarize-lm-head", action="store_true")
    p.add_argument("--attn-dropout", type=float, default=0.0)
    p.add_argument("--resid-dropout", type=float, default=0.0)
    p.add_argument("--attention-backend", choices=["sdpa", "flex", "flex-fa4"], default="sdpa", help="Attention implementation. flex-fa4 uses PyTorch FlexAttention with the FA4 FLASH backend when installed/supported.")
    p.add_argument("--no-flex-attention-compile", action="store_true", help="Disable torch.compile around FlexAttention/FA4.")
    p.add_argument("--flex-block-size", type=int, default=256, help="Block size used by FlexAttention causal block mask. flex-fa4 requires a multiple of 256; incompatible values are rounded up safely.")
    p.add_argument("--bitlinear-dx-grad", choices=["torch", "custom", "custom-quantized"], default="torch", help="dX gradient path. torch uses Tensor-Core matmul with cached FP16 dequantized ternary W; custom uses exact packed CUDA dX; custom-quantized uses approximate dp4a dX.")
    p.add_argument("--bitlinear-dweight-dtype", choices=["fp32", "tf32", "bf16", "fp16"], default="fp16", help="Matmul dtype for STE dW. fp16 is fastest on B200; fp32/tf32 is most conservative.")

    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--moe-num-layers", type=int, default=0, help="Number of transformer layers using MoE FFN")
    p.add_argument("--moe-layer-stride", type=int, default=1)
    p.add_argument("--moe-pad-to-multiple", type=int, default=16)
    p.add_argument("--router-aux-loss-coef", type=float, default=0.01)
    p.add_argument("--moe-routing-backend", choices=["naive", "grouped", "sonic"], default="grouped", help="MoE dispatch/routing implementation. sonic enables local SonicMoE-style grouped routing with optional tile rounding.")
    p.add_argument("--moe-topk-over-softmax", action="store_true", help="Use Qwen/Sonic-style topk(softmax(logits)) instead of softmax(topk(logits)).")
    p.add_argument("--no-moe-norm-topk-probs", action="store_true", help="Do not renormalize top-k probabilities when --moe-topk-over-softmax is used.")
    p.add_argument("--moe-token-rounding", action="store_true", help="Sonic-style tile-aware grouping. With the default 'pad' strategy this pads expert token groups to the tile size without dropping routed tokens.")
    p.add_argument("--moe-token-rounding-strategy", choices=["pad", "none"], default="pad")

    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--max-tokens", type=parse_count, default=None, help="Optional training token budget, e.g. 100M, 4B")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--cooldown-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--cooldown-start-frac", type=float, default=0.5)
    p.add_argument("--weight-decay-peak", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--use-custom-kernel", action="store_true", help="Kept for clarity; the custom kernel is enabled by default unless --no-custom-kernel is set")
    p.add_argument("--no-custom-kernel", action="store_true")
    p.add_argument("--allow-torch-fallback", action="store_true")
    p.add_argument("--custom-kernel-dweight-grad", action="store_true", help="Use the full extension scalar dW path for debugging. Default uses cached packed ternary weights plus PyTorch matmul/cuBLAS runtime for STE dW, which does not require cuBLAS development headers during extension compilation.")
    p.add_argument("--kernel-cu", default="EL_ternCUDA_kernel.cu")
    p.add_argument("--kernel-header", default="EL_ternCUDA_kernel.h")
    p.add_argument("--cuda-arch", default="auto", help="CUDA arch for the custom extension: auto/native, env, 10.0, 10.0+PTX, sm_100, etc. Default auto detects the visible GPU and overrides stale TORCH_CUDA_ARCH_LIST.")
    p.add_argument(
        "--extension-verbose",
        "--extension_verbose",
        "--verbose-extension",
        "--cuda-extension-verbose",
        "--kernel-verbose",
        dest="extension_verbose",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
        help="Show full PyTorch/ninja/nvcc extension build output. Accepts --extension-verbose or --extension-verbose=true.",
    )
    p.add_argument("--no-extension-progress", "--no_extension_progress", dest="no_extension_progress", action="store_true", help="Disable periodic CUDA extension build progress messages")
    p.add_argument("--extension-progress-interval", "--extension_progress_interval", dest="extension_progress_interval", type=float, default=2.0, help="Seconds between CUDA extension build progress updates")
    p.add_argument("--extension-build-dir", "--extension_build_dir", dest="extension_build_dir", default=None, help="Override the CUDA extension build directory; default is <output-dir>/cuda_extension_build_<abi>_<sm>, e.g. cuda_extension_build_headerless_streamhandle_v10_dxcache_fa4_flce_sonic_cudaaware_sm100_cu132 on B200")
    p.add_argument("--clean-extension-build", "--clean_extension_build", dest="clean_extension_build", action="store_true", help="Delete the extension build directory before compiling")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="reduce-overhead")
    p.add_argument("--no-flash-sdpa", action="store_true")
    p.add_argument("--no-liger", action="store_true")
    p.add_argument("--no-liger-fused-ce", action="store_true", help="Disable Liger fused linear cross entropy and materialize logits instead")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--ddp-find-unused-parameters", choices=["auto", "true", "false"], default="auto", help="DDP find_unused_parameters. auto=false for dense, true for MoE.")
    p.add_argument("--ddp-static-graph", action="store_true", help="Request DDP static_graph. Auto-disabled when no_sync/custom autograd/fused CE/FlexAttention would make it unsafe.")
    p.add_argument("--disable-ddp-no-sync", action="store_true", help="Synchronize gradients on every microbatch instead of using DDP no_sync during gradient accumulation.")
    p.add_argument("--no-ddp-gradient-as-bucket-view", action="store_true", help="Disable DDP gradient_as_bucket_view. Enabled by default to reduce gradient memory copies.")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=1000)
    args = p.parse_args()

    if args.stream_train and not (args.hf_dataset or args.data):
        p.error("--stream-train requires --hf-dataset or --data")
    if not args.stream_train and not args.token_cache and not args.hf_dataset and not args.data:
        p.error("provide --data, --hf-dataset, or an existing --token-cache")
    if args.stream_train and args.hf_dataset and not args.hf_streaming:
        # Direct stream training should be lazy by default.
        args.hf_streaming = True

    use_kernel = bool(not args.no_custom_kernel)
    target = parse_count(args.target_params)
    effective_flex_block_size = _effective_flex_block_size(args.attention_backend, args.flex_block_size)
    model_cfg = ModelConfig(
        vocab_size=0,
        seq_len=args.seq_len,
        architecture=args.architecture,
        target_params=target,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        intermediate_size=args.intermediate_size,
        multiple_of=args.multiple_of,
        activation=args.activation,
        tie_embeddings=not args.no_tie_embeddings,
        ternarize_lm_head=args.ternarize_lm_head,
        attn_dropout=args.attn_dropout,
        resid_dropout=args.resid_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        attention_backend=args.attention_backend,
        flex_attention_compile=not args.no_flex_attention_compile,
        flex_block_size=effective_flex_block_size,
        bitlinear_dx_grad=args.bitlinear_dx_grad,
        bitlinear_dweight_dtype=("fp32" if args.bitlinear_dweight_dtype == "tf32" else args.bitlinear_dweight_dtype),
        num_experts=args.num_experts,
        top_k=args.top_k,
        moe_num_layers=args.moe_num_layers,
        moe_layer_stride=args.moe_layer_stride,
        moe_pad_to_multiple=args.moe_pad_to_multiple,
        router_aux_loss_coef=args.router_aux_loss_coef,
        moe_routing_backend=args.moe_routing_backend,
        moe_topk_over_softmax=args.moe_topk_over_softmax,
        moe_norm_topk_probs=not args.no_moe_norm_topk_probs,
        moe_token_rounding=args.moe_token_rounding,
        moe_token_rounding_strategy=args.moe_token_rounding_strategy,
        use_custom_kernel=use_kernel,
        allow_torch_fallback=args.allow_torch_fallback or not use_kernel,
        use_torch_dweight_grad=not args.custom_kernel_dweight_grad,
    )
    train_cfg = TrainConfig(
        data=args.data or [],
        tokenizer=args.tokenizer,
        output_dir=args.output_dir,
        data_format=args.data_format,
        text_column=args.text_column,
        token_column=args.token_column,
        token_cache=args.token_cache,
        rebuild_token_cache=args.rebuild_token_cache,
        append_eos=not args.no_append_eos,
        tokenization_batch_size=args.tokenization_batch_size,
        parquet_batch_size=args.parquet_batch_size,
        token_cache_max_tokens=args.token_cache_max_tokens,
        vocab_size_override=args.vocab_size_override,
        hf_dataset=args.hf_dataset or [],
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        hf_streaming=bool(args.hf_streaming),
        stream_train=bool(args.stream_train),
        hf_data_files=args.hf_data_files,
        hf_cache_dir=args.hf_cache_dir,
        hf_revision=args.hf_revision,
        hf_token_env=args.hf_token_env,
        hf_trust_remote_code=args.hf_trust_remote_code,
        hf_shuffle_buffer=args.hf_shuffle_buffer,
        hf_skip=args.hf_skip,
        hf_take=args.hf_take,
        hf_interleave_probabilities=args.hf_interleave_probabilities,
        hf_ddp_shard_strategy=args.hf_ddp_shard_strategy,
        auto_tokenizer_fallback=args.auto_tokenizer_fallback,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        tokenizer_revision=args.tokenizer_revision,
        tokenizer_token_env=args.tokenizer_token_env,
        tokenizer_trust_remote_code=args.tokenizer_trust_remote_code,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        cooldown_lr=args.cooldown_lr,
        warmup_steps=args.warmup_steps,
        cooldown_start_frac=args.cooldown_start_frac,
        weight_decay_peak=args.weight_decay_peak,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        optimizer=args.optimizer,
        dtype=args.dtype,
        seed=args.seed,
        use_custom_kernel=use_kernel,
        allow_torch_fallback=args.allow_torch_fallback or not use_kernel,
        use_torch_dweight_grad=not args.custom_kernel_dweight_grad,
        bitlinear_dx_grad=args.bitlinear_dx_grad,
        bitlinear_dweight_dtype=("fp32" if args.bitlinear_dweight_dtype == "tf32" else args.bitlinear_dweight_dtype),
        attention_backend=args.attention_backend,
        flex_attention_compile=not args.no_flex_attention_compile,
        flex_block_size=effective_flex_block_size,
        kernel_cu=args.kernel_cu,
        kernel_header=args.kernel_header,
        cuda_arch=args.cuda_arch,
        extension_verbose=args.extension_verbose,
        extension_progress=not args.no_extension_progress,
        extension_progress_interval=args.extension_progress_interval,
        extension_build_dir=args.extension_build_dir,
        clean_extension_build=args.clean_extension_build,
        compile=args.compile,
        compile_mode=args.compile_mode,
        use_flash_sdpa=not args.no_flash_sdpa,
        use_liger=not args.no_liger,
        use_liger_fused_ce=not args.no_liger_fused_ce,
        fsdp=args.fsdp,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_static_graph=args.ddp_static_graph,
        ddp_no_sync=not args.disable_ddp_no_sync,
        ddp_gradient_as_bucket_view=not args.no_ddp_gradient_as_bucket_view,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )
    if model_cfg.architecture == "moe" and model_cfg.top_k > model_cfg.num_experts:
        raise ValueError("--top-k cannot exceed --num-experts")
    if model_cfg.architecture == "moe" and model_cfg.moe_num_layers <= 0:
        model_cfg.moe_num_layers = model_cfg.layers
    return model_cfg, train_cfg


def main() -> None:
    model_cfg, train_cfg = parse_args()
    try:
        train(model_cfg, train_cfg)
    finally:
        # Avoid NCCL resource-leak warnings after exceptions in torchrun jobs.
        cleanup_distributed()


if __name__ == "__main__":
    main()
