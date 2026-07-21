@echo off
setlocal
cd /d "%~dp0.."

echo [pi] Installing @earendil-works/pi-coding-agent globally...
call npm install -g --ignore-scripts @earendil-works/pi-coding-agent
if errorlevel 1 (
    echo [pi] npm install failed.
    exit /b 1
)

echo [pi] Deploying LMM Router provider config to %%USERPROFILE%%\.pi\agent ...
python helpers\pi_runner.py deploy
if errorlevel 1 (
    echo [pi] Config deploy failed.
    exit /b 1
)

echo.
echo [pi] Installed. Verify with:
echo   pi --version
echo   pi --list-models lmm
echo.
echo Interactive coding (utility slot):
echo   cd C:\path\to\your\project
echo   pi --model lmm-coding/utility
echo.
echo Chat slot via pi:
echo   pi --model lmm-router/chat
echo.
endlocal
