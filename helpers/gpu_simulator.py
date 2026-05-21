"""
GPU Simulator — simulate different GPU configurations for hardware planning.

Implements GPU simulation inspired by whichllm:
- Create synthetic GPU objects for testing
- Used for hardware planning: "what if I had an RTX 5090?"
- Support both predefined GPUs and custom VRAM specifications
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU specifications database (VRAM, bandwidth, compute capability)
# ---------------------------------------------------------------------------

_GPU_SPECS = {
    # NVIDIA RTX 50 series (2025)
    "RTX 5090": {
        "vram_gb": 32,
        "bandwidth_gbps": 1200,
        "compute_capability": 10.0,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 5080": {
        "vram_gb": 24,
        "bandwidth_gbps": 960,
        "compute_capability": 10.0,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 5070": {
        "vram_gb": 16,
        "bandwidth_gbps": 617,
        "compute_capability": 10.0,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 5060": {
        "vram_gb": 12,
        "bandwidth_gbps": 360,
        "compute_capability": 10.0,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    
    # NVIDIA RTX 40 series
    "RTX 4090": {
        "vram_gb": 24,
        "bandwidth_gbps": 1008,
        "compute_capability": 8.9,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 4080": {
        "vram_gb": 16,
        "bandwidth_gbps": 717,
        "compute_capability": 8.9,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 4070 Ti": {
        "vram_gb": 12,
        "bandwidth_gbps": 504,
        "compute_capability": 8.9,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 4070": {
        "vram_gb": 12,
        "bandwidth_gbps": 504,
        "compute_capability": 8.9,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 4060 Ti": {
        "vram_gb": 8,
        "bandwidth_gbps": 288,
        "compute_capability": 8.9,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 4060": {
        "vram_gb": 8,
        "bandwidth_gbps": 360,
        "compute_capability": 8.9,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    
    # NVIDIA RTX 30 series
    "RTX 3090 Ti": {
        "vram_gb": 24,
        "bandwidth_gbps": 1008,
        "compute_capability": 8.6,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 3090": {
        "vram_gb": 24,
        "bandwidth_gbps": 936,
        "compute_capability": 8.6,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 3080 Ti": {
        "vram_gb": 12,
        "bandwidth_gbps": 912,
        "compute_capability": 8.6,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 3080": {
        "vram_gb": 10,
        "bandwidth_gbps": 760,
        "compute_capability": 8.6,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 3070 Ti": {
        "vram_gb": 8,
        "bandwidth_gbps": 608,
        "compute_capability": 8.6,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 3070": {
        "vram_gb": 8,
        "bandwidth_gbps": 448,
        "compute_capability": 8.6,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "RTX 3060": {
        "vram_gb": 12,
        "bandwidth_gbps": 360,
        "compute_capability": 8.6,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    
    # NVIDIA Tesla/Datacenter
    "A100": {
        "vram_gb": 40,
        "bandwidth_gbps": 1555,
        "compute_capability": 8.0,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "A100 80GB": {
        "vram_gb": 80,
        "bandwidth_gbps": 2039,
        "compute_capability": 8.0,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "H100": {
        "vram_gb": 80,
        "bandwidth_gbps": 3350,
        "compute_capability": 9.0,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    "H200": {
        "vram_gb": 141,
        "bandwidth_gbps": 4096,
        "compute_capability": 9.0,
        "vendor": "NVIDIA",
        "discrete": True,
    },
    
    # AMD Radeon
    "RX 7900 XTX": {
        "vram_gb": 24,
        "bandwidth_gbps": 960,
        "compute_capability": 7.0,
        "vendor": "AMD",
        "discrete": True,
    },
    "RX 7900 XT": {
        "vram_gb": 20,
        "bandwidth_gbps": 800,
        "compute_capability": 7.0,
        "vendor": "AMD",
        "discrete": True,
    },
    "RX 6900 XT": {
        "vram_gb": 16,
        "bandwidth_gbps": 512,
        "compute_capability": 6.0,
        "vendor": "AMD",
        "discrete": True,
    },
    "RX 6800 XT": {
        "vram_gb": 16,
        "bandwidth_gbps": 512,
        "compute_capability": 6.0,
        "vendor": "AMD",
        "discrete": True,
    },
    
    # Apple Silicon (unified memory)
    "M1 Ultra": {
        "vram_gb": 128,
        "bandwidth_gbps": 800,
        "compute_capability": 5.0,
        "vendor": "Apple",
        "discrete": False,
        "unified_memory": True,
    },
    "M2 Ultra": {
        "vram_gb": 192,
        "bandwidth_gbps": 800,
        "compute_capability": 5.5,
        "vendor": "Apple",
        "discrete": False,
        "unified_memory": True,
    },
    "M3 Max": {
        "vram_gb": 128,
        "bandwidth_gbps": 400,
        "compute_capability": 6.0,
        "vendor": "Apple",
        "discrete": False,
        "unified_memory": True,
    },
    "M4 Max": {
        "vram_gb": 128,
        "bandwidth_gbps": 500,
        "compute_capability": 6.5,
        "vendor": "Apple",
        "discrete": False,
        "unified_memory": True,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_gpu(gpu_name: str, vram_gb: Optional[float] = None) -> Dict[str, Any]:
    """
    Create a synthetic GPU object for simulation.
    
    Used for hardware planning: "what if I had an RTX 5090?"
    
    Args:
        gpu_name: GPU model name (e.g., "RTX 4090")
        vram_gb: Optional VRAM override in GB (for custom configurations)
    
    Returns:
        dict with GPU specifications (vram_gb, bandwidth_gbps, compute_capability, etc.)
    """
    # Normalize GPU name
    gpu_name_normalized = gpu_name.strip()
    
    # Try to find exact match
    if gpu_name_normalized in _GPU_SPECS:
        specs = _GPU_SPECS[gpu_name_normalized].copy()
        if vram_gb is not None:
            specs["vram_gb"] = vram_gb
        return specs
    
    # Try case-insensitive match
    gpu_name_lower = gpu_name_normalized.lower()
    for key, specs in _GPU_SPECS.items():
        if key.lower() == gpu_name_lower:
            specs_copy = specs.copy()
            if vram_gb is not None:
                specs_copy["vram_gb"] = vram_gb
            return specs_copy
    
    # If not found, create a custom GPU with the given VRAM
    # Use conservative defaults
    if vram_gb is None:
        vram_gb = 24.0  # Default to 24GB if not specified
    
    # Estimate bandwidth based on VRAM (rough heuristic)
    if vram_gb >= 80:
        bandwidth_gbps = 2000
    elif vram_gb >= 32:
        bandwidth_gbps = 1000
    elif vram_gb >= 24:
        bandwidth_gbps = 800
    elif vram_gb >= 16:
        bandwidth_gbps = 500
    elif vram_gb >= 12:
        bandwidth_gbps = 400
    else:
        bandwidth_gbps = 300
    
    return {
        "vram_gb": vram_gb,
        "bandwidth_gbps": bandwidth_gbps,
        "compute_capability": 8.0,  # Conservative default
        "vendor": "Unknown",
        "discrete": True,
        "name": gpu_name_normalized,
    }


def list_available_gpus() -> list[str]:
    """
    Get list of available GPU names for simulation.
    """
    return sorted(_GPU_SPECS.keys())


def get_gpu_specs(gpu_name: str) -> Optional[Dict[str, Any]]:
    """
    Get specifications for a specific GPU.
    Returns None if GPU not found.
    """
    return _GPU_SPECS.get(gpu_name)


def compare_gpus(gpu_name_1: str, gpu_name_2: str) -> Dict[str, Any]:
    """
    Compare two GPUs and return their differences.
    
    Returns:
        dict with:
            gpu1: specs for first GPU
            gpu2: specs for second GPU
            vram_diff: VRAM difference (GB)
            bandwidth_diff: Bandwidth difference (GB/s)
            bandwidth_ratio: gpu2 bandwidth / gpu1 bandwidth
    """
    gpu1 = simulate_gpu(gpu_name_1)
    gpu2 = simulate_gpu(gpu_name_2)
    
    vram_diff = gpu2["vram_gb"] - gpu1["vram_gb"]
    bandwidth_diff = gpu2["bandwidth_gbps"] - gpu1["bandwidth_gbps"]
    bandwidth_ratio = gpu2["bandwidth_gbps"] / gpu1["bandwidth_gbps"] if gpu1["bandwidth_gbps"] > 0 else 0
    
    return {
        "gpu1": gpu1,
        "gpu2": gpu2,
        "vram_diff": vram_diff,
        "bandwidth_diff": bandwidth_diff,
        "bandwidth_ratio": round(bandwidth_ratio, 2),
    }
