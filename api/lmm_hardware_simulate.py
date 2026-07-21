"""
LmmHardwareSimulate API — GPU simulation endpoint for hardware planning.

POST /api/plugins/a0_lmm_router/hardware/simulate
POST /api/plugins/a0_lmm_router/hardware/compare
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from flask import Request
from helpers.api import ApiHandler

# Add plugin directory to path for imports
plugin_dir = Path(__file__).parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from helpers.gpu_simulator import simulate_gpu, list_available_gpus, compare_gpus  # type: ignore
from helpers.speed_estimator import estimate_tok_per_sec  # type: ignore
from helpers.context_calculator import estimate_vram_detailed  # type: ignore

logger = logging.getLogger(__name__)


class LmmHardwareSimulate(ApiHandler):
    """POST /api/plugins/a0_lmm_router/hardware/simulate - Simulate GPU and show model compatibility"""

    async def process(self, input: dict, request: Request) -> dict:
        """Simulate a GPU configuration and show which models fit and their estimated speed."""
        try:
            data = input or {}
            gpu_name = data.get("gpu_name", "")
            vram_gb = data.get("vram_gb", 0)
            ctx_size = data.get("ctx_size", 8192)
            top_n = data.get("top_n", 20)

            if not gpu_name:
                return {
                    "ok": False,
                    "error": "gpu_name is required",
                }

            # Get simulated GPU
            gpu = simulate_gpu(gpu_name, vram_gb if vram_gb > 0 else None)

            # Get installed models
            try:
                from tools.lmm_host_helper import _load_manifest, _get_models_dir_from_env, _get_env_path
                
                env_path = _get_env_path(str(plugin_dir / "docker" / "docker-compose.lmm.yml"))
                models_dir = _get_models_dir_from_env(env_path)
                manifest = _load_manifest(models_dir)
                models = manifest.get("models", {})
            except Exception as e:
                logger.warning(f"Failed to load installed models: {e}")
                return {
                    "ok": False,
                    "error": f"Failed to load installed models: {e}",
                }

            # Calculate VRAM and speed for each model
            results = []
            for model_id, model_info in models.items():
                try:
                    # Get model file path
                    model_path = str(Path(models_dir) / model_info.get("path", "") / model_info.get("file", ""))
                    
                    # Estimate VRAM
                    vram_estimate = estimate_vram_detailed(model_path, ctx_size)
                    
                    # Determine fit type
                    fits = vram_estimate["total_gb"] <= gpu["vram_gb"]
                    if fits:
                        fit_type = "full_gpu"
                    elif vram_estimate["weights_gb"] <= gpu["vram_gb"]:
                        fit_type = "partial_offload"
                    else:
                        fit_type = "cpu_only"

                    # Estimate speed
                    quantization = "Q4_K_M"  # Default, could extract from filename
                    speed_estimate = estimate_tok_per_sec(
                        gpu_name=gpu["name"] if "name" in gpu else gpu_name,
                        model_size_gb=model_info.get("size_gb", 0),
                        quantization=quantization,
                        ctx_size=ctx_size,
                        fit_type=fit_type,
                    )

                    results.append({
                        "model_id": model_id,
                        "file": model_info.get("file", ""),
                        "size_gb": model_info.get("size_gb", 0),
                        "role_hint": model_info.get("role_hint", ""),
                        "fits": fits,
                        "vram_needed_gb": vram_estimate["total_gb"],
                        "vram_breakdown": vram_estimate,
                        "estimated_tok_per_sec": speed_estimate["estimated_tok_per_sec"],
                        "speed_confidence": speed_estimate["confidence"],
                        "speed_range": speed_estimate["confidence_range"],
                        "fit_type": fit_type,
                        "benchmark_score": model_info.get("benchmark_score", 0),
                        "task_profiles": model_info.get("task_profiles", []),
                    })
                except Exception as e:
                    logger.warning(f"Failed to estimate for model {model_id}: {e}")
                    continue

            # Sort: fitting models first, then by benchmark score, then by speed
            results.sort(key=lambda x: (
                -x["fits"],  # fitting models first
                -x["benchmark_score"],  # higher benchmark score first
                -x["estimated_tok_per_sec"],  # faster models first
            ))

            return {
                "ok": True,
                "gpu": gpu,
                "gpu_name": gpu_name,
                "ctx_size": ctx_size,
                "total_models": len(models),
                "fitting_models": sum(1 for r in results if r["fits"]),
                "models": results[:top_n],
            }

        except Exception as e:
            logger.exception("Error in hardware simulation")
            return {
                "ok": False,
                "error": str(e),
            }


class LmmHardwareCompare(ApiHandler):
    """POST /api/plugins/a0_lmm_router/hardware/compare - Compare two GPUs"""

    async def process(self, input: dict, request: Request) -> dict:
        """Compare two GPUs and show the differences."""
        try:
            data = input or {}
            gpu_name_1 = data.get("gpu_name_1", "")
            gpu_name_2 = data.get("gpu_name_2", "")

            if not gpu_name_1 or not gpu_name_2:
                return {
                    "ok": False,
                    "error": "Both gpu_name_1 and gpu_name_2 are required",
                }

            comparison = compare_gpus(gpu_name_1, gpu_name_2)

            return {
                "ok": True,
                "comparison": comparison,
            }

        except Exception as e:
            logger.exception("Error in GPU comparison")
            return {
                "ok": False,
                "error": str(e),
            }


class LmmHardwareListGpus(ApiHandler):
    """GET /api/plugins/a0_lmm_router/hardware/list-gpus - List available GPUs for simulation"""

    async def process(self, input: dict, request: Request) -> dict:
        """Get list of available GPU names for simulation."""
        try:
            gpus = list_available_gpus()
            return {
                "ok": True,
                "gpus": gpus,
                "count": len(gpus),
            }
        except Exception as e:
            logger.exception("Error listing GPUs")
            return {
                "ok": False,
                "error": str(e),
            }
