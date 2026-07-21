# LMM Router Plugin & MCP Server - Bug Report

**Date:** 2026-05-16
**Plugin Version:** 1.3.0
**Status:** Issues documented, awaiting external fixes

---

## Summary

Multiple issues identified in the LMM Router plugin and MCP server integration that prevent containers from starting and GUI from receiving messages.

---

## Critical Issues

### 1. BAT File Syntax Error - `lmm_manager.bat`

**File:** `c:/Users/frant/agent-zero/agent-zero-2/lmm_manager.bat`
**Lines:** 361-383 (Health Checks section)

**Issue:** 
- Original code used `curl` command which is not available or has syntax issues on Windows
- Error message: `: was unexpected at this time.`
- This causes the entire status command to fail

**Current State:**
- Health checks have been temporarily disabled with a note
- Container status checks work, but health endpoint checks are skipped

**Impact:** 
- Users cannot verify if containers are actually healthy
- Status command completes but shows incomplete information

---

### 2. Host Helper Startup - Invalid Arguments

**File:** `c:/Users/frant/agent-zero/agent-zero-2/start_agent_zero.bat`
**Line:** 126

**Issue:**
- BAT file passes `--models-dir` and `--env-file` flags to host helper
- Host helper (`lmm_host_helper.py`) only accepts: `--port`, `--compose`, `--project-dir`
- Extra arguments cause host helper to fail on startup

**Current State:**
- Arguments have been removed (temporary fix applied)
- Host helper starts successfully

**Impact:**
- Without fix, host helper fails to start
- No communication between container and host
- GUI messaging completely broken

---

### 3. PROJECT_DIR Variable Empty

**File:** `c:/Users/frant/agent-zero/agent-zero-2/start_agent_zero.bat`
**Line:** 19

**Issue:**
- `set PROJECT_DIR=%~dp0` results in empty PROJECT_DIR variable
- Debug output shows: `Debug: Project directory:` (empty)
- Causes all relative path resolutions to fail

**Symptoms:**
- LMM compose file not found: `[WARNING]  not found in C:\Users\frant\agent-zero\agent-zero-2\`
- Docker containers fail to start
- Agent Zero container fails to start

**Impact:**
- Complete startup failure
- No containers can be launched

**Note:** Temporary fix applied to remove trailing backslash, but root cause needs investigation

---

### 4. LMM Docker Containers Not Starting

**File:** `c:/Users/frant/agent-zero/agent-zero-2/start_agent_zero.bat`
**Lines:** 92-106

**Issue:**
- Docker compose command fails due to PROJECT_DIR being empty
- Error: `[WARNING]  not found in C:\Users\frant\agent-zero\agent-zero-2\`
- Compose file path: `usr\plugins\a0_lmm_router\docker\docker-compose.lmm.yml`

**Impact:**
- No llama.cpp containers (chat, utility, embed) start
- Model services completely unavailable

---

### 5. Agent Zero Container Startup Failure

**File:** `c:/Users/frant/agent-zero/agent-zero-2/start_agent_zero.bat`
**Lines:** 156-175

**Issue:**
- Container exists but fails to start
- Error: `[ERROR] Failed to start existing Agent Zero container`
- Script then marks Docker as unavailable and skips further operations

**Impact:**
- Main Agent Zero container not running
- No web UI available at http://localhost:5080
- MCP server cannot start inside container

---

### 6. MCP Server Launcher Not Found Inside Container

**File:** `c:/Users/frant/agent-zero/agent-zero-2/start_agent_zero.bat`
**Lines:** 186-192

**Issue:**
- Script checks for `/a0/usr/plugins/a0_lmm_router/launcher.py` inside container
- File not found: `[SKIP] a0_lmm_router launcher not found inside container`
- Launcher exists on host but not mounted into container

**Impact:**
- MCP server cannot be started from within container
- Plugin's MCP integration completely non-functional

**Root Cause:** 
- Docker volume mount configuration may be missing or incorrect
- Plugin directory not properly mounted into container at `/a0/usr/plugins/`

---

### 7. MCP Server Module Path Mismatch

**File:** `c:/Users/frant/agent-zero/agent-zero-2/start_agent_zero.bat`
**Lines:** 199-208

**Issue:**
- Script expects MCP server at: `/a0/usr/plugins/a0_lmm_router/mcp_server/server.py`
- Host path: `usr/plugins/a0_lmm_router/mcp_server/server.py`
- Path mismatch suggests incorrect volume mount configuration

**Impact:**
- MCP server cannot start even if launcher is found
- Plugin's MCP features completely unavailable

---

## GUI Messaging Issues

### 8. Host Helper Communication Failure

**Files:** 
- `usr/plugins/a0_lmm_router/helpers/fleet_models.py` (line 26)
- `usr/plugins/a0_lmm_router/webui/config.html` (lines 285-320)

**Issue:**
- GUI tries to communicate with host helper at `http://host.docker.internal:55501`
- If host helper fails to start (see Issue #2), all GUI operations fail
- Token file path: `/a0/tmp/lmm_host_token` (container) vs `%TEMP%\a0_lmm_host.key` (host)

**Symptoms:**
- GUI shows "manager offline" status
- No slot status updates
- Model management buttons non-functional
- Fleet status unavailable

**Impact:**
- Complete GUI paralysis
- Users cannot manage containers or models through web interface

---

### 9. API Endpoint Routing

**Files:**
- `usr/plugins/a0_lmm_router/api/fleet_status.py`
- `usr/plugins/a0_lmm_router/api/llamacpp_status.py`

**Issue:**
- APIs depend on host helper being reachable
- If host helper down, all API calls fail with `_router_unreachable: true`
- No fallback mechanism or error handling for unreachable helper

**Impact:**
- All plugin API endpoints fail
- No status information available
- Cannot diagnose issues through API

---

## Configuration Issues

### 10. Docker Compose Environment File

**File:** `usr/plugins/a0_lmm_router/docker/docker-compose.lmm.env`

**Issue:**
- Environment variables reference model paths that may not exist
- No validation that model files are present before starting containers
- Containers start but fail if models missing

**Model Paths in Config:**
- Chat/Utility: `C:/Users/frant/A0-Data-Permanent/A0_v.adam/models/utility/qwen3.5_9b/Qwen3.5-9B-Q4_K_M.gguf`
- Embedding: `C:/Users/frant/A0-Data-Permanent/A0_v.adam/models/embedding/nomic/nomic-embed-text-v1.5.Q4_K_M.gguf`

**Status:** Both paths verified to exist on host

---

### 11. Docker Network Configuration

**File:** `c:/Users/frant/agent-zero/agent-zero-2/start_agent_zero.bat`
**Lines:** 78-90

**Issue:**
- Script creates `a0-lmm-net` network if missing
- No validation that network is properly configured
- No check if containers can actually communicate on this network

**Impact:**
- Containers may start but cannot communicate
- Host-to-container communication may fail
- MCP server unreachable from container

---

## MCP Server Specific Issues

### 12. MCP Server Not Running

**File:** `usr/plugins/a0_lmm_router/mcp_server/server.py`

**Issue:**
- MCP server exists on host but is not running
- Startup script tries to launch it inside container (Issue #6, #7)
- No standalone MCP server startup on host

**Impact:**
- MCP features completely unavailable
- No Model Context Protocol integration
- Plugin cannot be used as MCP server

---

### 13. MCP Server Port Configuration

**Expected Port:** 8095 (from start_agent_zero.bat line 204)

**Issue:**
- No validation that port 8095 is available
- No check if port is already in use
- No error handling if port bind fails

**Impact:**
- Silent failures if port conflict
- MCP server may fail to start without clear error message

---

## Dependency Issues

### 14. Python Path Configuration

**File:** `c:/Users/frant/agent-zero/agent-zero-2/start_agent_zero.bat`
**Line:** 21

**Issue:**
- `set PYTHONPATH=%PROJECT_DIR%` relies on PROJECT_DIR being set correctly
- If PROJECT_DIR empty, PYTHONPATH is empty
- Python imports may fail

**Impact:**
- Helper scripts may fail to import required modules
- Unclear error messages if imports fail

---

### 15. Helper Module Imports

**File:** `usr/plugins/a0_lmm_router/tools/lmm_host_helper.py`
**Lines:** 50-58

**Issue:**
- Host helper adds helpers directory to sys.path
- Imports `context_calculator` which may have its own dependencies
- If dependencies missing, host helper fails silently

**Impact:**
- Context window calculation may fail
- Model assignment may use incorrect context sizes
- Unclear error messages

---

## Documentation Issues

### 16. Outdated Documentation

**File:** `usr/plugins/a0_lmm_router/README.md`

**Issue:**
- Documentation may not reflect MCP server changes
- May reference old startup procedures
- May not document new MCP architecture

**Impact:**
- Users follow outdated instructions
- Confusion about correct startup procedure
- Difficulty troubleshooting issues

---

## Temporary Fixes Applied (For Reference)

The following temporary fixes have been applied but should be replaced with proper solutions:

1. **lmm_manager.bat:** Health checks disabled (lines 357-363)
2. **start_agent_zero.bat:** Invalid host helper arguments removed (line 126)
3. **start_agent_zero.bat:** Trailing backslash removal added (lines 20-21)

These are workarounds and do not address root causes.

---

## Recommended Investigation Order

1. Fix PROJECT_DIR variable resolution (Issue #3)
2. Verify Docker volume mounts for plugin directory (Issue #6, #7)
3. Fix host helper argument parsing (Issue #2)
4. Restore proper health checks in lmm_manager.bat (Issue #1)
5. Add MCP server startup validation (Issue #12, #13)
6. Improve error handling throughout (Issue #9, #15)
7. Update documentation to reflect MCP architecture (Issue #16)

---

## Environment Details

- **OS:** Windows
- **Docker:** Docker Desktop (presumed)
- **Python:** Available in PATH
- **Plugin Path:** `c:/Users/frant/agent-zero/agent-zero-2/usr/plugins/a0_lmm_router/`
- **Models Path:** `C:/Users/frant/A0-Data-Permanent/A0_v.adam/models/`
- **Host Helper Port:** 55501
- **MCP Server Port:** 8095
- **LMM Container Ports:** 8080 (chat), 8088 (utility), 8082 (embed)
