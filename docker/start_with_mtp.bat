@echo off
:: ============================================================
:: Start Agent Zero LMM fleet WITH MTP (Multi-Token Prediction)
:: Reference: https://github.com/ggml-org/llama.cpp/pull/22673
:: Requires: Docker Desktop, NVIDIA Container Toolkit in WSL2
::           llama.cpp build containing PR #22673
::           (merged 2026-05-16, commit 10829dbc). The published
::           ghcr.io/ggml-org/llama.cpp:server-cuda image may not
::           yet include it — if --spec-type draft-mtp is rejected,
::           build a custom image from llama.cpp master.
::
:: MTP requires an MTP-capable model in the UTILITY slot
:: (Qwen3.6-27B, Qwen3.6-35BA3B-MoE, DeepSeek V3/R1). Configure
:: UTILITY_MODEL_PATH in docker-compose.lmm.env accordingly.
:: ============================================================
setlocal

set COMPOSE_FILE=%~dp0docker-compose.lmm.yml
set MTP_FILE=%~dp0docker-compose.lmm.mtp.yml
set ENV_FILE=%~dp0docker-compose.lmm.env

echo.
echo === Agent Zero LMM Fleet — MTP Mode (PR #22673) ===
echo CHAT slot:    port 8080 (Gemma-4)
echo UTILITY slot: port 8088 (MTP-capable model + --spec-type draft-mtp ~1.85x decode)
echo EMBED slot:   port 8082 (nomic-embed)
echo.

:: Ensure the Docker network exists
docker network create a0-lmm-net 2>nul

:: Pull latest image — MTP requires a build containing PR #22673
echo Pulling latest llama.cpp CUDA image...
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda

echo.
echo Starting fleet with MTP enabled on utility slot...
docker compose ^
    -f "%COMPOSE_FILE%" ^
    -f "%MTP_FILE%" ^
    --env-file "%ENV_FILE%" ^
    up -d

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Fleet is up. Endpoints:
    echo   Chat endpoint:    http://localhost:8080/v1  (model: chat)
    echo   Utility endpoint: http://localhost:8088/v1  (model: utility, MTP enabled)
    echo   Embed endpoint:   http://localhost:8082/v1  (model: embedding)
    echo.
    echo In Hermes: switch to preset "a0-chat" or "a0-utility"
    echo In Agent Zero: already configured via a0_lmm_router plugin
) else (
    echo ERROR: docker compose failed.
    echo Check: Docker Desktop is running, WSL2 is active, NVIDIA drivers are installed.
)

pause
