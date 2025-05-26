#!/bin/bash
set -e

docker build -t opencv_cpp_dev -f ./Dockerfile .

docker run -it \
  --name opencv-cpp-cuda_env \
  --gpus all \
  --restart unless-stopped \
  opencv_cpp_dev

# To exec into the container later:
# docker exec -it opencv-cpp-cuda_env /bin/bash