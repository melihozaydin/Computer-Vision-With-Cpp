

----
# Sources

## CUDA
* https://nvidia.github.io/container-wiki/toolkit/container-images.html#using-non-cuda-images
* https://catalog.ngc.nvidia.com/collections

## Docker
- use base image 
* https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container
 ,
----

# Computer Vision with C++ and CUDA

## Overview
This project aims to provide a development environment for computer vision applications using OpenCV and CUDA. It utilizes Docker to create a containerized setup that simplifies the installation and management of dependencies.

## Project Structure
```
Computer-Vision-With-Cpp
├── 00 - Setup
│   └── 00-BuildSystem
│       └── 02-Containerized
│           └── Opencv_CUDA_Docker
│               ├── deploy.bat
│               └── Dockerfile
├── src
├── include
└── README.md
```

## Setup Instructions

### Prerequisites
- Docker must be installed on your machine.
- Ensure that your system supports GPU and that the necessary drivers are installed.

### Building the Docker Image
Navigate to the `00 - Setup/00-BuildSystem/02-Containerized/Opencv_CUDA_Docker` directory and run the following command to build the Docker image:

```bash
docker build -t opencv_cpp_dev -f Dockerfile .
```

### Running the Docker Container
After building the image, you can run the Docker container with GPU support using the following command:

```bash
docker run -it \
  --name opencv-cpp-cuda_env \
  --gpus all \
  --restart unless-stopped \
  opencv_cpp_dev
```

### Accessing the Container
To access the running container, you can use the following command:

```bash
docker exec -it opencv_cpp_cuda_env /bin/bash
```

## Usage
Once inside the container, you can start developing your computer vision applications using OpenCV and CUDA. Place your source code in the `src` directory and any header files in the `include` directory.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. Your contributions are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for more details.