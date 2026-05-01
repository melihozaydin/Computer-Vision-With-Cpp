@echo off
SETLOCAL

docker build -t opencv_cpp_dev -f .\Dockerfile .

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..\..\..\..") do set "REPO_ROOT=%%~fI"

echo Repo root: %REPO_ROOT%

docker rm -f opencv-cpp-cuda_env >nul 2>nul

docker run -it ^
  --name opencv-cpp-cuda_env ^
  --gpus all ^
  --restart unless-stopped ^
  -v "%REPO_ROOT%:/workspace" ^
  -w "/workspace/04 - Opencv/CUDA" ^
  opencv_cpp_dev ^
  /bin/bash

::docker exec -it opencv_cpp_dev /bin/bash

ENDLOCAL