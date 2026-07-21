@echo off
:: ============================================================
:: Start Agent Zero LMM fleet WITH MTP (Qwen3.5 Multi-Token Prediction)
:: Requires: Docker Desktop, NVIDIA Container Toolkit in WSL2
::           llama.cpp image post-May-16-2026 (built after PR #22673)
:: ============================================================
setlocal

set COMPOSE_FILE=%~dp0docker-compose.lmm.yml
set MTP_FILE=%~dp0docker-compose.lmm.mtp.yml
set ENV_FILE=%~dp0docker-compose.lmm.env

echo.
echo === Agent Zero LMM Fleet — MTP Mode ===
echo CHAT slot:    port 8080 (Gemma-4)
echo UTILITY slot: port 8088 (Qwen3.5 + MTP ~1.4x faster)
echo EMBED slot:   port 8082 (nomic-embed)
echo.

:: Ensure the Docker network exists
docker network create a0-lmm-net 2>nul

:: Pull latest image (post-PR #22673 has MTP support)
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
