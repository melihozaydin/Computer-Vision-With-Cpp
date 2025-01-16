# https://github.com/lanzani/opencv-cuda-docker
ARG CUDA="11.3.1"
# ARG CUDA="12.6.1"
ARG UBUNTU="20.04"
ARG ARCH=""

# https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda${ARCH}:${CUDA}-devel-ubuntu${UBUNTU}

ENV DEBIAN_FRONTEND=noninteractive

# See https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${NVIDIA_DISTRO}/3bf863cc.pub


RUN apt update && apt install -y --no-install-recommends \
    # Build tools for compiling C/C++ code
    build-essential pkg-config \
    cmake git gcc g++ ninja-build \
    # Debugger for debugging code
    gdb \
    # Utilities for downloading and extracting files
    wget unzip yasm \
    # Package manager for installing software without a package manager (e.g. Ubuntu's `dpkg`)
    checkinstall \
    ### OpenCV Dependencies ###
    # Development libraries for IEEE 1394 cameras (DC-1394)
    libdc1394-22 libdc1394-22-dev \
    # Linear algebra library optimized for performance
    libatlas-base-dev \
    # Fortran compiler (for numerical computations)
    gfortran \
    # Flann development libraries (for efficient nearest neighbor search)
    libflann-dev \
    # Thread building blocks library (for concurrent programming)
    libtbb2 \
    libtbb-dev \
    # JPEG image compression and decompression library
    libjpeg-dev \
    # PNG image compression and decompression library
    libpng-dev \
    # TIFF image compression and decompression library
    libtiff-dev \
    libtiff5-dev \
    # OpenGL Extension Wrangler (GLEW) development libraries
    libglew-dev \
    # Compression library (zlib)
    zlib1g-dev \
    # Geospatial data abstraction library (GDAL) development libraries
    libgdal-dev \
    # Flags and gflags development libraries (for command-line argument parsing)
    libgflags-dev \
    libgoogle-glog-dev \
    # Protocol Buffers compiler and development libraries
    libprotobuf-dev \
    protobuf-compiler \
    # Video encoding and decoding library
    ffmpeg \
    # AVFoundation development libraries for video encoding and decoding
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    # Leptonica image processing library development libraries
    libleptonica-dev \
    # Tesseract OCR engine development libraries
    libtesseract-dev \
    # GTK+ GUI toolkit development libraries (for GUI applications)
    libgtk-3-dev libgtk2.0-dev \
    # VTK (Visualization Toolkit) development libraries for 3D visualization
    libvtk6-dev \
    # Linear algebra library (LAPACK) development libraries
    liblapack-dev \
    # Development libraries for v4l (video4linux)
    libv4l-dev \
    doxygen \
    # Eigen
    libeigen3-dev \
    # Python
    python-dev \
    python-numpy \
    python3-dev \
    python3-numpy \
    # HDF5 serial library development libraries
    libhdf5-serial-dev

# Build OpenCV
WORKDIR /tmp
RUN wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV}.zip && unzip ${OPENCV}.zip && rm ${OPENCV}.zip
RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV}.zip && unzip ${OPENCV}.zip && rm ${OPENCV}.zip

# This needs to be a single "RUN" command as new run commands return back to Workdir.
# You can also keep changing Workdirs but "RUN cd ..." and "WORKDIR ..." behave differently. 
RUN mkdir opencv-${OPENCV}/build && cd opencv-${OPENCV}/build && \
    # https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html#tutorial_config_reference_general_cpu
    cmake -GNinja \
        # Specify build configuration (Ninja generator)
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV}/modules \
        # Optimize OpenCV using fast math libraries
        -DENABLE_FAST_MATH=ON \
        # CUBLAS support
        -DWITH_CUBLAS=ON \
        # Allow non-free OpenCV modules to be built and installed
        -DOPENCV_ENABLE_NONFREE=ON \
        # GStreamer support
        -DWITH_GSTREAMER=OFF \
        # Specify the build configuration type (RELEASE means a typical, non-debug build)
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        # Disable building and running tests
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        # Disable building examples
        -DBUILD_EXAMPLES=OFF \
        # Generate pkg-config for C++ linking
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        #https://jayrambhia.com/blog/opencv-with-tbb
        #-D WITH_TBB=ON \
        .. && \
    ninja && \
    ninja install && \
    ldconfig


# Cleanup
RUN rm -rf /tmp/* && rm -rf /var/lib/apt/lists/*

# Test opencv Cpp install
## https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/
RUN pkg-config --modversion opencv4
# Test opencv cuda support
COPY Test .
RUN ls -a
RUN cmake build . && make

# Test opencv python cuda install
RUN python3 -c "import cv2; print(cv2.__version__)"
#RUN python3 cudaDeviceInfo.py

## TODO
#RUN g++ test_opencv_cuda test_opencv_cuda.cpp -o test_opencv_cuda_1 `pkg-config --cflags  --libs` 
#RUN ./test_opencv_cuda_1

# Install Eigen
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends libeigen3-dev

# Test Eigen3
#RUN pkg-config --modversion opencv4

