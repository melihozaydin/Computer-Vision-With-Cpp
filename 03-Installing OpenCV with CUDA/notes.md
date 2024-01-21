# How To Install and Build OpenCV C++ with NVIDIA CUDA GPU in Visual Studio Code

1. Installing the source files from OpenCV's GitHub repository, including the OpenCV contrib module.
   1. <https://github.com/opencv/opencv>
   2. <https://github.com/opencv/opencv_contrib>
2. Installing Visual Studio Code with the necessary development tools for C++.
3. Installing CUDA and cuDNN from NVIDIA's website.
   """
   https://developer.nvidia.com/cuda-downloads
   https://developer.nvidia.com/rdp/cudnn-download
   https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
   """
   1. Go to NVIDIA's Developer website: <https://developer.nvidia.com/>
   2. Sign in or create a new account.
   3. Navigate to the CUDA Toolkit page: <https://developer.nvidia.com/cuda-toolkit>
   4. Download the CUDA Toolkit installer for your operating system and architecture. Choose the version compatible with your GPU and operating system.
   5. Once the download is complete, run the CUDA Toolkit installer and follow the on-screen instructions to install CUDA on your system.
   6.  After installing CUDA, navigate to the cuDNN page: <https://developer.nvidia.com/cudnn>
   7.  Download the cuDNN library for your version of CUDA. Make sure to select the correct version that matches your installed CUDA version.
   8.  Extract the downloaded cuDNN package to a directory of your choice.
   9.  Copy the extracted files to the corresponding CUDA installation directory. Typically, you need to copy the contents of the `bin`, `include`, and `lib` directories to the corresponding directories in your CUDA installation.
   10. Set the necessary environment variables for CUDA and cuDNN. Add the CUDA and cuDNN paths to your system's PATH environment variable. Additionally, add the CUDA and cuDNN library paths to the LIBRARY_PATH (on Linux) or LIB (on Windows) environment variable.

4.  Installing CMake, which will be used to build the source files.
5.  Configuring OpenCV with GPU support in CMake.
   1. Open CMake GUI.
   2. Specify the source files location by browsing and selecting the folder where the OpenCV source files are downloaded and extracted.
   3. Specify the build directory by browsing and selecting or creating a folder where the build files will be generated.
   4. Click on the "Configure" button.
   5. Choose the version of Visual Studio you have installed.
   6. Select the architecture of your computer (e.g., x64).
   7. Click on the "Finish" button.
   8. Scroll down to find the "WITH_CUDA" option and check it to enable CUDA support.
   9. Scroll down further to find the "OPENCV_EXTRA_MODULES_PATH" option.
   10. Set the value of "OPENCV_EXTRA_MODULES_PATH" to the location of the OpenCV contrib module for GPU support that you downloaded.
   11. Click on the "Configure" button again to update the configuration.
   12. If there are no errors, click on the "Generate" button.
   13. Once the generation process is complete, close CMake GUI.
     After configuring OpenCV with GPU support using CMake, you can proceed to build the source files and then install OpenCV with GPU support.

6.  Building the source files using CMake.
   1. Open the command prompt or terminal.
   2. Change the directory to the build directory where the CMake files were generated.
   3. Run the following command to build the source files:

      ```
      cmake --build . --config Release
      ```

      This command builds the source files in Release mode. You can also use `--config Debug` for building in Debug mode.
   4. Wait for the build process to complete. It may take some time depending on the hardware and the size of the project.
   5. Once the build process is finished, the binaries will be generated in the specified build directory.

   After building the source files, you can proceed with the installation of OpenCV with GPU support.

7.  Setting up and testing OpenCV with GPU support in Visual Studio Code.

To set up and test OpenCV with GPU support in Visual Studio Code, follow these steps:

1. Open Visual Studio Code.
2. Create a new folder for your project or open an existing project.
3. Install the CMake Tools extension in Visual Studio Code if you haven't already.
4. Open the Command Palette in Visual Studio Code (press `Ctrl + Shift + P` or `Cmd + Shift + P`).
5. Search for and select "CMake: Configure" to configure the CMake project.
6. Select the target architecture and configuration (e.g., x64 and Release) if prompted.
7. Once the configuration is complete, open the Command Palette again and search for and select "CMake: Build" to build the project.
8. Wait for the build process to finish.

After the build process is complete, you can test OpenCV with GPU support by running a sample program. Here's an example program to print GPU device information:

```cpp
#include <opencv2/core.hpp>
#include <opencv2/cudaxcore.hpp>
#include <iostream>

int main() {
    try {
        cv::cuda::printCudaDeviceInfo(0);
    } catch (const cv::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
```

Save this program with a `.cpp` file extension in your project folder. Then, open the Command Palette again and search for and select "CMake: Run Without Debugging" to run the program.

If everything is set up correctly, you should see the GPU device information printed in the output console. This indicates that OpenCV with GPU support is working correctly in Visual Studio Code.

Note: Make sure you have the necessary GPU drivers and CUDA/CUDNN installed on your system for GPU support in OpenCV.
