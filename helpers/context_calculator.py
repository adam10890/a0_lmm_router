"""
helpers/context_calculator.py — Automatic context window sizing.

Calculates optimal context window size based on:
1. Model's max context from GGUF metadata (n_ctx_train)
2. Available VRAM (dynamic calculation)
3. Model size (smaller model = larger context)

KV cache formula: ctx_size * 2 * n_layer * n_embd * bytes_per_token
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Optional

log = logging.getLogger("a0_lmm_router.context_calculator")

# Default fallback values if metadata reading fails
_DEFAULT_CTX_SIZES = {
    "chat": 32768,
    "utility": 16384,
    "embedding": 8192,
    "vision": 32768,
    "reasoning": 32768,
}

# Safety margin for VRAM (GB)
_VRAM_SAFETY_MARGIN_GB = 2.0


def read_gguf_metadata(model_path: str) -> dict:
    """
    Read GGUF metadata to extract context-related information.

    Returns dict with:
        n_ctx_train: max training context length
        n_embd: embedding dimension
        n_layer: number of layers
        n_head: number of attention heads
        file_size_gb: model file size in GB
    """
    path = Path(model_path)
    if not path.exists():
        log.warning(f"Model file not found: {model_path}")
        return {}

    file_size_gb = path.stat().st_size / (1024**3)

    try:
        # Simple GGUF reader - reads key-value pairs from the file
        # GGUF format: magic (4 bytes) + version (4 bytes) + tensor_count (8 bytes) + KV count (8 bytes)
        # Then KV pairs: key_len (8 bytes) + key_str + value_type (1 byte) + value_data

        metadata = {
            "n_ctx_train": None,
            "n_embd": None,
            "n_layer": None,
            "n_head": None,
            "file_size_gb": file_size_gb,
        }

        with open(path, "rb") as f:
            # Read header
            magic = f.read(4)
            if magic != b"GGUF":
                log.warning(f"Not a GGUF file: {model_path}")
                return metadata

            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            # Read KV pairs
            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="ignore")
                value_type = struct.unpack("<B", f.read(1))[0]

                # Read value based on type
                if value_type == 0:  # UINT8
                    value = struct.unpack("<B", f.read(1))[0]
                elif value_type == 1:  # INT8
                    value = struct.unpack("<b", f.read(1))[0]
                elif value_type == 2:  # UINT16
                    value = struct.unpack("<H", f.read(2))[0]
                elif value_type == 3:  # INT16
                    value = struct.unpack("<h", f.read(2))[0]
                elif value_type == 4:  # UINT32
                    value = struct.unpack("<I", f.read(4))[0]
                elif value_type == 5:  # INT32
                    value = struct.unpack("<i", f.read(4))[0]
                elif value_type == 6:  # FLOAT32
                    value = struct.unpack("<f", f.read(4))[0]
                elif value_type == 7:  # BOOL
                    value = struct.unpack("<?", f.read(1))[0]
                elif value_type == 8:  # STRING
                    str_len = struct.unpack("<Q", f.read(8))[0]
                    value = f.read(str_len).decode("utf-8", errors="ignore")
                elif value_type == 9:  # ARRAY
                    arr_type = struct.unpack("<B", f.read(1))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    if arr_type == 4:  # UINT32 array
                        value = [struct.unpack("<I", f.read(4))[0] for _ in range(arr_len)]
                    elif arr_type == 5:  # INT32 array
                        value = [struct.unpack("<i", f.read(4))[0] for _ in range(arr_len)]
                    elif arr_type == 6:  # FLOAT32 array
                        value = [struct.unpack("<f", f.read(4))[0] for _ in range(arr_len)]
                    else:
                        # Skip unknown array types
                        for _ in range(arr_len):
                            if arr_type == 4:
                                f.read(4)
                            elif arr_type == 5:
                                f.read(4)
                            elif arr_type == 6:
                                f.read(4)
                        value = None
                else:
                    # Skip unknown types
                    continue

                # Extract relevant metadata
                if key == "n_ctx_train":
                    metadata["n_ctx_train"] = int(value) if isinstance(value, (int, str)) else None
                elif key == "n_embd":
                    metadata["n_embd"] = int(value) if isinstance(value, (int, str)) else None
                elif key == "n_layer":
                    metadata["n_layer"] = int(value) if isinstance(value, (int, str)) else None
                elif key == "n_head":
                    metadata["n_head"] = int(value) if isinstance(value, (int, str)) else None

        return metadata

    except Exception as e:
        log.error(f"Failed to read GGUF metadata from {model_path}: {e}")
        return {"file_size_gb": file_size_gb}


def calculate_kv_cache_size(
    ctx_size: int,
    n_layer: int,
    n_embd: int,
    bytes_per_token: int = 2,
) -> float:
    """
    Calculate KV cache size in GB for a given context window.

    Formula: ctx_size * 2 * n_layer * n_embd * bytes_per_token
    The *2 is for K and V matrices.
    """
    if not n_layer or not n_embd:
        return 0.0

    total_bytes = ctx_size * 2 * n_layer * n_embd * bytes_per_token
    return total_bytes / (1024**3)


def calculate_optimal_context(
    model_path: str,
    slot: str,
    available_vram_gb: float,
    other_slots_vram_gb: float = 0.0,
) -> dict:
    """
    Calculate optimal context window size for a model.

    Args:
        model_path: Path to the GGUF model file
        slot: Slot name (chat, utility, embedding, vision, reasoning)
        available_vram_gb: Total GPU VRAM available in GB
        other_slots_vram_gb: VRAM used by other running slots in GB

    Returns:
        dict with:
            recommended_ctx: int - recommended context size
            n_ctx_train: int - max context from model metadata
            vram_for_kv: float - VRAM needed for KV cache
            vram_for_weights: float - VRAM for model weights
            total_vram_needed: float - total VRAM needed
            reasoning: str - explanation of the calculation
    """
    metadata = read_gguf_metadata(model_path)
    file_size_gb = metadata.get("file_size_gb", 0)
    n_ctx_train = metadata.get("n_ctx_train")
    n_layer = metadata.get("n_layer")
    n_embd = metadata.get("n_embd")

    # Estimate VRAM for weights (file size * 1.15 for overhead)
    vram_for_weights = file_size_gb * 1.15

    # VRAM available for this slot (total - other slots - safety margin)
    vram_for_slot = available_vram_gb - other_slots_vram_gb - _VRAM_SAFETY_MARGIN_GB
    if vram_for_slot < vram_for_weights:
        # Not enough VRAM even for weights
        return {
            "recommended_ctx": _DEFAULT_CTX_SIZES.get(slot, 8192),
            "n_ctx_train": n_ctx_train,
            "vram_for_kv": 0.0,
            "vram_for_weights": vram_for_weights,
            "total_vram_needed": vram_for_weights,
            "reasoning": f"Insufficient VRAM: {vram_for_slot:.1f}GB available, {vram_for_weights:.1f}GB needed for weights. Using default context.",
        }

    # VRAM available for KV cache
    vram_for_kv = vram_for_slot - vram_for_weights

    # If we have n_layer and n_embd, calculate max context from VRAM
    if n_layer and n_embd and vram_for_kv > 0:
        # Solve for ctx_size: ctx_size = vram_for_kv * 1024^3 / (2 * n_layer * n_embd * bytes_per_token)
        bytes_per_token = 2  # FP16 KV cache
        max_ctx_from_vram = int(
            (vram_for_kv * (1024**3)) / (2 * n_layer * n_embd * bytes_per_token)
        )

        # Round down to power of 2 multiple for efficiency
        def round_to_power_of_2(x):
            if x <= 0:
                return 2048
            power = 1
            while power * 2 <= x:
                power *= 2
            return power

        max_ctx_from_vram = round_to_power_of_2(max_ctx_from_vram)

        # Apply model's max context limit
        if n_ctx_train:
            recommended_ctx = min(max_ctx_from_vram, n_ctx_train)
            reasoning = (
                f"Model supports {n_ctx_train} tokens. "
                f"VRAM allows {max_ctx_from_vram} tokens. "
                f"Using {recommended_ctx} tokens."
            )
        else:
            recommended_ctx = max_ctx_from_vram
            reasoning = (
                f"Model max context unknown. "
                f"VRAM allows {max_ctx_from_vram} tokens. "
                f"Using {recommended_ctx} tokens."
            )
    else:
        # Fallback: use default context size
        recommended_ctx = _DEFAULT_CTX_SIZES.get(slot, 8192)
        reasoning = (
            f"Could not calculate from metadata. "
            f"Using default context of {recommended_ctx} tokens."
        )

    # Calculate actual KV cache size for recommended context
    if n_layer and n_embd:
        actual_kv_size = calculate_kv_cache_size(recommended_ctx, n_layer, n_embd)
    else:
        # Rough estimate: 0.5 GB per 8K context per 10GB model
        actual_kv_size = (recommended_ctx / 8192) * (file_size_gb / 10) * 0.5

    total_vram_needed = vram_for_weights + actual_kv_size

    return {
        "recommended_ctx": recommended_ctx,
        "n_ctx_train": n_ctx_train,
        "vram_for_kv": actual_kv_size,
        "vram_for_weights": vram_for_weights,
        "total_vram_needed": total_vram_needed,
        "reasoning": reasoning,
    }
