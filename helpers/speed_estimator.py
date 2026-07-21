"""
Speed Estimator — estimate tokens/sec for LLM inference.

Implements speed estimation inspired by whichllm:
- GPU bandwidth-based estimation
- Quantization efficiency factors
- Model size and context considerations
- Fit type adjustments (full GPU vs partial offload vs CPU-only)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU bandwidth lookup table (GB/s)
# Based on theoretical memory bandwidth of popular GPUs
# ---------------------------------------------------------------------------

_GPU_BANDWIDTH_GBPS = {
    # NVIDIA RTX 50 series (2025)
    "RTX 5090": 1200,
    "RTX 5080": 960,
    "RTX 5070": 617,
    "RTX 5060": 360,
    
    # NVIDIA RTX 40 series
    "RTX 4090": 1008,
    "RTX 4080": 717,
    "RTX 4070 Ti": 504,
    "RTX 4070": 504,
    "RTX 4060 Ti": 288,
    "RTX 4060": 360,
    
    # NVIDIA RTX 30 series
    "RTX 3090 Ti": 1008,
    "RTX 3090": 936,
    "RTX 3080 Ti": 912,
    "RTX 3080": 760,
    "RTX 3070 Ti": 608,
    "RTX 3070": 448,
    "RTX 3060": 360,
    
    # NVIDIA RTX 20 series
    "RTX 2080 Ti": 616,
    "RTX 2080": 448,
    "RTX 2070": 448,
    "RTX 2060": 336,
    
    # NVIDIA GTX 16 series
    "GTX 1660 Ti": 288,
    "GTX 1660": 192,
    
    # NVIDIA Tesla/Datacenter
    "A100": 1555,  # 40GB
    "A100 80GB": 2039,
    "H100": 3350,
    "H200": 4096,
    
    # AMD Radeon
    "RX 7900 XTX": 960,
    "RX 7900 XT": 800,
    "RX 6900 XT": 512,
    "RX 6800 XT": 512,
    
    # Apple Silicon (unified memory bandwidth)
    "M1 Ultra": 800,
    "M2 Ultra": 800,
    "M3 Max": 400,
    "M4 Max": 500,
}


# ---------------------------------------------------------------------------
# Quantization bytes per parameter (approximate)
# ---------------------------------------------------------------------------

_QUANT_BYTES = {
    "Q2_K": 2.0,
    "Q3_K": 3.0,
    "Q3_K_S": 3.0,
    "Q3_K_M": 3.0,
    "Q3_K_L": 3.0,
    "Q4_0": 4.0,
    "Q4_K": 4.5,
    "Q4_K_S": 4.5,
    "Q4_K_M": 4.5,
    "Q5_0": 5.0,
    "Q5_K": 5.5,
    "Q5_K_S": 5.5,
    "Q5_K_M": 5.5,
    "Q6_K": 6.0,
    "Q8_0": 8.0,
    "F16": 16.0,
    "F32": 32.0,
}


# ---------------------------------------------------------------------------
# Fit type multipliers (adjust for full GPU vs partial vs CPU)
# ---------------------------------------------------------------------------

_FIT_MULTIPLIERS = {
    "full_gpu": 1.0,
    "partial_offload": 0.6,
    "cpu_only": 0.15,
}


# ---------------------------------------------------------------------------
# Context size impact (larger context = slightly slower due to KV cache access)
# ---------------------------------------------------------------------------

def _context_impact(ctx_size: int) -> float:
    """
    Returns a multiplier for context size impact on speed.
    
    Larger contexts have more KV cache to access, which slightly slows down generation.
    - 4K context: 1.0 (baseline)
    - 8K context: 0.95
    - 16K context: 0.90
    - 32K context: 0.85
    - 64K context: 0.80
    """
    if ctx_size <= 4096:
        return 1.0
    elif ctx_size <= 8192:
        return 0.95
    elif ctx_size <= 16384:
        return 0.90
    elif ctx_size <= 32768:
        return 0.85
    else:
        return 0.80


# ---------------------------------------------------------------------------
# Extract quantization from filename
# ---------------------------------------------------------------------------

def _extract_quantization(filename: str) -> str:
    """
    Extract quantization type from GGUF filename.
    
    Examples:
    - "model.Q4_K_M.gguf" -> "Q4_K_M"
    - "model-q4_k_m.gguf" -> "Q4_K_M"
    """
    import re
    
    # Try common patterns
    patterns = [
        r"\.([Qq][0-9]_[Kk]_[SM])\.gguf",
        r"\.([Qq][0-9]_[Kk])\.gguf",
        r"\.([Qq][0-9]_[0-9])\.gguf",
        r"-([Qq][0-9]_[Kk]_[SM])-",
        r"-([Qq][0-9]_[Kk])-",
        r"_([Qq][0-9]_[Kk]_[SM])_",
        r"_([Qq][0-9]_[Kk])_",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            quant = match.group(1).upper()
            # Normalize common variations
            if quant in _QUANT_BYTES:
                return quant
    
    # Default to Q4_K_M if unknown
    return "Q4_K_M"


# ---------------------------------------------------------------------------
# Extract GPU name from hardware info
# ---------------------------------------------------------------------------

def _extract_gpu_name(gpu_info: Dict[str, Any]) -> str:
    """
    Extract GPU name from hardware info dict.
    
    Handles various formats from hardware_inspector.py.
    """
    name = gpu_info.get("name", "")
    if not name:
        return ""
    
    # Normalize name
    name_upper = name.upper()
    
    # Try to match against our lookup table
    for gpu_key in _GPU_BANDWIDTH_GBPS:
        if gpu_key.upper() in name_upper:
            return gpu_key
    
    # Return original name if no match
    return name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_tok_per_sec(
    gpu_name: str,
    model_size_gb: float,
    quantization: str = "Q4_K_M",
    ctx_size: int = 8192,
    fit_type: str = "full_gpu",
) -> Dict[str, Any]:
    """
    Estimate tokens/sec based on GPU bandwidth and model characteristics.
    
    Args:
        gpu_name: GPU model name (e.g., "RTX 4090")
        model_size_gb: Model file size in GB
        quantization: Quantization type (e.g., "Q4_K_M")
        ctx_size: Context window size in tokens
        fit_type: "full_gpu" / "partial_offload" / "cpu_only"
    
    Returns:
        dict with:
            estimated_tok_per_sec: float - estimated tokens per second
            confidence_range: tuple - (min, max) confidence range
            confidence: str - "high" / "medium" / "low"
            gpu_bandwidth_gbps: float - GPU bandwidth used
            bytes_per_param: float - bytes per parameter
    """
    # Get GPU bandwidth
    gpu_bandwidth = _GPU_BANDWIDTH_GBPS.get(gpu_name, 500)  # default 500 GB/s
    
    # Get quantization bytes per parameter
    bytes_per_param = _QUANT_BYTES.get(quantization, 4.5)  # default Q4_K_M
    
    # Get fit type multiplier
    fit_multiplier = _FIT_MULTIPLIERS.get(fit_type, 1.0)
    
    # Get context impact
    context_factor = _context_impact(ctx_size)
    
    # Estimate tokens/sec
    # Formula: bandwidth / (model_size * bytes_per_param)
    # Adjusted by fit type and context size
    base_tok_per_sec = gpu_bandwidth / (model_size_gb * bytes_per_param)
    estimated_tok_per_sec = base_tok_per_sec * fit_multiplier * context_factor
    
    # Confidence range (±30% for estimation uncertainty)
    # More uncertainty for partial offload and CPU-only
    uncertainty = 0.30 if fit_type == "full_gpu" else 0.50 if fit_type == "partial_offload" else 0.70
    confidence_range = (
        round(estimated_tok_per_sec * (1 - uncertainty), 1),
        round(estimated_tok_per_sec * (1 + uncertainty), 1),
    )
    
    # Confidence level
    if fit_type == "full_gpu" and gpu_name in _GPU_BANDWIDTH_GBPS:
        confidence = "high"
    elif fit_type == "partial_offload":
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "estimated_tok_per_sec": round(estimated_tok_per_sec, 1),
        "confidence_range": confidence_range,
        "confidence": confidence,
        "gpu_bandwidth_gbps": gpu_bandwidth,
        "bytes_per_param": bytes_per_param,
    }


def estimate_from_hardware(
    gpu_info: Dict[str, Any],
    model_size_gb: float,
    filename: str = "",
    ctx_size: int = 8192,
    available_vram_gb: float = 0,
) -> Dict[str, Any]:
    """
    Estimate speed from hardware info dict (from hardware_inspector.py).
    
    Args:
        gpu_info: GPU info dict from hardware_inspector
        model_size_gb: Model file size in GB
        filename: Model filename (to extract quantization)
        ctx_size: Context window size
        available_vram_gb: Available VRAM (to determine fit type)
    
    Returns:
        Same as estimate_tok_per_sec, with additional fit_type determination
    """
    # Extract GPU name
    gpu_name = _extract_gpu_name(gpu_info)
    
    # Extract quantization from filename
    if filename:
        quantization = _extract_quantization(filename)
    else:
        quantization = "Q4_K_M"
    
    # Determine fit type
    gpu_vram_gb = gpu_info.get("total_vram_mb", 0) / 1024.0
    if gpu_vram_gb == 0:
        fit_type = "cpu_only"
    elif available_vram_gb > 0:
        if model_size_gb * 1.1 < available_vram_gb:
            fit_type = "full_gpu"
        else:
            fit_type = "partial_offload"
    else:
        # Can't determine, assume full GPU if discrete
        fit_type = "full_gpu" if gpu_info.get("discrete") else "cpu_only"
    
    result = estimate_tok_per_sec(
        gpu_name=gpu_name,
        model_size_gb=model_size_gb,
        quantization=quantization,
        ctx_size=ctx_size,
        fit_type=fit_type,
    )
    
    result["fit_type"] = fit_type
    result["quantization"] = quantization
    
    return result


def get_all_gpu_bandwidths() -> Dict[str, float]:
    """
    Get all GPU bandwidth values (for debugging/testing).
    """
    return _GPU_BANDWIDTH_GBPS.copy()
