@echo off
SETLOCAL

docker build -t opencv_cpp_dev -f .\Dockerfile .

docker run -it ^
  --name opencv-cpp-cuda_env ^
  --gpus all ^
  --restart unless-stopped ^
  opencv_cpp_dev

::docker exec -it opencv_cpp_dev /bin/bash

ENDLOCAL