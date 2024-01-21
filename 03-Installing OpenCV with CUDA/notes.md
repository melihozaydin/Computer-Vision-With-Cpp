# How To Install and Build OpenCV C++ with NVIDIA CUDA GPU in Visual Studio Code

1. Downloading the source files from OpenCV's GitHub repository, including the OpenCV contrib module.
   1. <https://github.com/opencv/opencv>
   2. <https://github.com/opencv/opencv_contrib>
2. Installing Visual Studio Code with the necessary development tools for C++.
   - https://visualstudio.microsoft.com/downloads/
   - Select Desktop dev with c++
3. Installing CUDA and cuDNN from NVIDIA's website.
   https://developer.nvidia.com/cuda-downloads
   https://developer.nvidia.com/rdp/cudnn-download
   https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
   1. Go to NVIDIA's Developer website: <https://developer.nvidia.com/>
   2. Sign in or create a new account.
   3. Navigate to the CUDA Toolkit page: <https://developer.nvidia.com/cuda-toolkit>
   4. Download the CUDA Toolkit installer for your operating system and architecture. Choose the version compatible with your GPU and operating system.
   5. Once the download is complete, run the CUDA Toolkit installer and follow the on-screen instructions to install CUDA on your system.
   6.  After installing CUDA, navigate to the cuDNN page: <https://developer.nvidia.com/cudnn>
   7.  Download the cuDNN library for your version of CUDA. Make sure to select the correct version that matches your installed CUDA version.
   8.  Extract the downloaded cuDNN package to a directory of your choice.
   9.  Copy the extracted files to the corresponding CUDA installation directory. Typically, you need to copy the contents of the `bin`, `include`, and `lib` directories to the corresponding directories in your CUDA installation.

4.  Instal CMake, which will be used to build the source files.
5.  Configuring OpenCV with GPU support in CMake.
    1. Open CMake GUI.
    2. Specify the source files location by browsing and selecting the folder where the OpenCV source files are downloaded and extracted.
    3. Specify the build directory by browsing and selecting or creating a folder where the build files will be generated.
    4. Click on the "Configure" button.
    5.  Choose the version of Visual Studio you have installed.
    6.  Select the architecture of your computer (e.g., x64).
    7.  Click on the "Finish" button.

    8.  Check Below
      - WITH_CUDA
      - opencv_world
      - ENABLE_FAST_MATH
    9.  Set the value of "OPENCV_EXTRA_MODULES_PATH" to the location of the OpenCV contrib module for GPU support that you downloaded.
    10. Click on the "Configure" button again to update the configuration.
    11. Check CUDA_FAST_MATH
    12. Specify "CUDA_ARCH_BIN"
       - https://en.wikipedia.org/wiki/CUDA#GPUs_supported (8.6 for 3090)
    13. If there are no errors, click on the "Generate" button.
    14. Once the generation process is complete, close CMake GUI.

6.  Building the source files using CMake.
    1. Open the command prompt or terminal.
    2. Change the directory to the build directory where the CMake files were generated.
    3. Run the following command to build the source files:
      ```
      cmake --build . --target INSTALL --config Release
      ```
      This command builds the source files in Release mode. You can also use `--config Debug` for building in Debug mode.
    4. Wait for the build process to complete. It may take some time depending on the hardware and the size of the project.
    5. Once the build process is finished, the binaries will be generated in the specified build directory.

7.  Setting up and testing OpenCV with GPU support in Visual Studio Code.
    1. Open Visual Studio Code.
    2. Create a new folder for your project or open an existing project.
    3.  Install the CMake Tools extension in Visual Studio Code if you haven't already.
    4.  Open the Command Palette in Visual Studio Code (press `Ctrl + Shift + P` or `Cmd + Shift + P`).
    5.  Search for and select "CMake: Configure" to configure the CMake project.
    6.  Select the target architecture and configuration (e.g., x64 and Release) if prompted.
    7.  Once the configuration is complete, open the Command Palette again and search for and select "CMake: Build" to build the project.
    8.  Wait for the build process to finish.

- After the build is done:
  - Save the "build/directory/install" folder to not have to compile again.
  - Also you can move it to a seperate folder and get rid of everything else
  - Add "install_dir/x64/vc17/" bin and lib folder paths to "PATH" in enviroment variables
  - After that cmake should auto find it with : find_package( OpenCV REQUIRED )
  - Note: If vscode was open prior to installation cmake may not find it.
    - Restart vscode to resolve.

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
