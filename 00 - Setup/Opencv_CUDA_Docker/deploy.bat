@echo off
SETLOCAL

::podman build -t cpp_opencv4_cuda .

::podman build -t opencv_cpp_dev -f .\Dockerfile

podman run -it ^
  --name opencv-cpp-cuda_env ^
  --gpus all ^
  --restart unless-stopped ^
  opencv_cpp_dev

::podman exec -it opencv_cpp_dev /bin/bash

ENDLOCAL