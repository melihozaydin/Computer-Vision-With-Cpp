@echo off
SETLOCAL

::podman build -t cpp_opencv4_cuda .

podman run -d ^
  --name opencv_cpp_cuda ^
  --device nvidia.com/gpu=all ^
  --restart unless-stopped ^
  cpp_opencv4_cuda

podman exec -it cpp_opencv4_cuda
ENDLOCAL