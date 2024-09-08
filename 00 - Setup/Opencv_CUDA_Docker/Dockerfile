# Stage 0: Base Image with CUDA and OpenCV 4
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Update package index and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libprotobuf-dev libleveldb-dev libsnappy-dev openjdk-14-jdk \ 
    libgflags-dev libgoogle-glog-dev libsysinfo-api-perl libatlas-base-dev libblas-dev liblapack-dev \ 
    libboost-all-dev libeigen3-dev libprotoc15 protobuf-compiler

# Install OpenCV 4 with CUDA
RUN mkdir /tmp/opencv && \
    cd /tmp/opencv && \
    wget https://github.com/opencv/opencv/archive/refs/tags/4.5.3.zip && \
    unzip 4.5.3.zip && \
    mv opencv-4.5.3/ opencv-4.5.3-cuda11.3.1 && \
    cd /tmp/opencv/opencv-4.5.3-cuda11.3.1 && \
    mkdir release && cd release && \
    cmake -D CMAKE_BUILD_TYPE=Release -D OPENCV_EXTRA_MODULES=boost -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda .. && \
    make -j$(nproc) && \
    make install

# Move OpenCV to the correct location
RUN mkdir /opt/opencv && \
    mv /usr/local/lib/libopencv_*.so* /opt/opencv/

# Set up environment variables for OpenCV
ENV LD_LIBRARY_PATH=/opt/opencv/lib:$LD_LIBRARY_PATH

# Install Eigen
RUN apt-get update && \
    apt-get install -y --no-install-recommends libeigen3-dev