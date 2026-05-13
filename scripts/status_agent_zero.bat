@echo off
title Agent Zero - Service Status
color 0B

echo.
echo  ============================================================
echo               AGENT ZERO - SERVICE STATUS
echo  ============================================================
echo.

echo  Service Health Check:
echo  ------------------------------------------------------------

REM PostgreSQL
docker exec agent-zero-pgvector pg_isready -U agent_zero >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] PostgreSQL        : RUNNING  - localhost:5433
) else (
    echo   [!!] PostgreSQL        : STOPPED
)

REM LMM Docker Containers
echo.
echo   LMM Docker Containers (llama.cpp):
echo   ------------------------------------------------------------
docker ps --filter "name=a0-llama-chat" --format "{{.Status}}" 2>nul | findstr /i "Up" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] a0-llama-chat     : RUNNING
) else (
    echo   [!!] a0-llama-chat     : STOPPED
)
docker ps --filter "name=a0-llama-utility" --format "{{.Status}}" 2>nul | findstr /i "Up" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] a0-llama-utility  : RUNNING
) else (
    echo   [--] a0-llama-utility  : DOWN  (profile: full)
)
docker ps --filter "name=a0-llama-embed" --format "{{.Status}}" 2>nul | findstr /i "Up" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] a0-llama-embed    : RUNNING
) else (
    echo   [--] a0-llama-embed    : DOWN  (profile: full)
)

echo.
echo   LMM Health Checks:
echo   ------------------------------------------------------------

REM llama.cpp Chat Model (Magistral 12B)
curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Chat Model        : HEALTHY  - http://localhost:8080 (Magistral 12B)
) else (
    echo   [!!] Chat Model        : UNREACHABLE  - port 8080
)

REM llama.cpp Utility Model (Phi-3.5)
curl -s http://localhost:8088/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Utility Model     : HEALTHY  - http://localhost:8088 (Phi-3.5 CPU)
) else (
    echo   [!!] Utility Model     : UNREACHABLE  - port 8088
)

REM llama.cpp Embedding Model (Nomic)
curl -s http://localhost:8082/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Embedding Model   : HEALTHY  - http://localhost:8082 (Nomic)
) else (
    echo   [!!] Embedding Model   : UNREACHABLE  - port 8082
)

REM WebSocket
netstat -an | findstr :8890 | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] WebSocket Server  : RUNNING  - ws://localhost:8890
) else (
    echo   [!!] WebSocket Server  : STOPPED
)

REM Agent Zero Web UI
curl -s http://localhost:50001 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Agent Zero UI     : RUNNING  - http://localhost:50001
) else (
    echo   [!!] Agent Zero UI     : STOPPED
)

echo.
echo  ------------------------------------------------------------
echo.

REM GPU Status
echo  GPU Status:
echo  ------------------------------------------------------------

echo.
echo  ------------------------------------------------------------
echo  Commands:
echo    ..\..\..\..\..\..\start_agent_zero.bat  - Start all services (run from repo root)
echo    ..\..\..\..\..\..\stop_agent_zero.bat   - Stop all services (run from repo root)
echo    ..\..\..\..\..\..\lmm_manager.bat       - LMM container manager (run from repo root)
echo  ------------------------------------------------------------
echo  Note: Run these scripts from the agent-zero-2 repo root directory.
echo.

pause
