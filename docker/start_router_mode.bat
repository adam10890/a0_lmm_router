@echo off
:: ============================================================
:: Start Agent Zero LMM fleet in Router Mode
:: (single container, all models, hot-swap on demand)
::
:: This is the ALTERNATIVE to start_agent_zero.bat's 3-slot fleet.
:: Run one or the other — never both (they share port 8080).
::
:: To switch back to 3-slot mode:
::   1. stop_router_mode.bat
::   2. start_agent_zero.bat   (or equivalent docker compose -f docker-compose.lmm.yml up -d)
:: ============================================================
setlocal

set HERE=%~dp0
set COMPOSE_FILE=%HERE%docker-compose.lmm.router.yml
set ENV_FILE=%HERE%docker-compose.lmm.env

echo.
echo === Agent Zero LMM Fleet — Router Mode ===
echo Single endpoint: http://localhost:8080
echo Aliases routed via preset.ini:
echo   "model": "chat"      -> gemma-4-E4B-uncensored
echo   "model": "utility"   -> Qwen3.5-9B
echo   "model": "embedding" -> nomic-embed v1.5
echo.

:: Ensure the Docker network exists (created by start_agent_zero.bat
:: normally, but make this script self-sufficient when run standalone).
docker network create a0-lmm-net 2>nul

:: Refuse to run if the 3-slot fleet is up — port collision would crash us.
for %%C in (a0-llama-chat a0-llama-utility a0-llama-embed) do (
    docker ps --filter "name=%%C" --format "{{.Names}}" 2>nul | findstr /i "%%C" >nul && (
        echo [ERROR] %%C is running. Router Mode shares port 8080 with the 3-slot fleet.
        echo         Stop it first:
        echo           docker compose -f "%HERE%docker-compose.lmm.yml" --env-file "%ENV_FILE%" down
        exit /b 1
    )
)

echo Starting Router Mode container...
docker compose ^
    -f "%COMPOSE_FILE%" ^
    --env-file "%ENV_FILE%" ^
    up -d

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Router container started. Waiting for /health...
    powershell -NoProfile -Command "$ok=$false; 1..20 | %% { try { $r = Invoke-WebRequest -Uri 'http://localhost:8080/health' -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop; if ($r.StatusCode -eq 200) { Write-Host ('[OK] Router responding after ' + ($_ * 3) + 's'); $ok=$true; break } } catch { Start-Sleep -Seconds 3 } }; if (-not $ok) { Write-Host '[WARN] Router not responding after 60s. Check: docker logs a0-llama-router' }"
    echo.
    echo Registered models:
    powershell -NoProfile -Command "try { (Invoke-RestMethod -Uri 'http://localhost:8080/v1/models' -TimeoutSec 5).data | ForEach-Object { Write-Host ('  - ' + $_.id) } } catch { Write-Host '  (could not query /v1/models — check the router logs)' }"
) else (
    echo ERROR: docker compose failed.
    echo Check: Docker Desktop is running, WSL2 is active, NVIDIA drivers installed.
)

echo.
pause
