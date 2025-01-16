
https://www.simonwenkel.com/notes/software_libraries/opencv/compiling-opencv.html#the-problem


- Complier installation:
  - https://code.visualstudio.com/docs/cpp/config-mingw
    - "Found OpenCV Windows Pack but it has no binaries compatible with your configuration
      ```
      https://stackoverflow.com/questions/70942644/how-to-fix-warning-found-opencv-windows-pack-but-it-has-no-binaries-compatible
      
      As the error suggests, CMake found your OpenCV installation, but it is not compatible. What is it not compatible with? Your compiler. The OpenCV installation is built with MSVC 15 (it also includes a 14 build). You have asked CMake to use MinGW as your compiler. Libraries need to have been built with the same (well, with some leeway) compiler for everything to work.

      You have two options:

          Build OpenCV yourself with MinGW, or try a third-party MinGW binary distribution of OpenCV. Web searches for "opencv mingw" turn up some possibilities.
          Use the MSVC compiler for your project. Microsoft offers some free versions of its tools. Just be sure to install the optional older toolsets so that you have access to the VC 15 tools to match OpenCV.
      ```
      - https://medium.com/csmadeeasy/opencv-c-installation-on-windows-with-mingw-c0fc1499f39
  
  - https://code.visualstudio.com/docs/cpp/config-msvc 
    - The CXX compiler identification is unknown
      ```
          https://stackoverflow.com/questions/20632860/the-cxx-compiler-identification-is-unknown
          I was getting the terminal output:

          The C compiler identification is unknown
          The CXX compiler identification is unknown

          I checked the CMakeError.log output:

          \build\CMakeFiles\CMakeError.log

          It showed the error:

          warning MSB8003: The WindowsSDKDir property is not defined. Some build tools may not be found.

          Going back to visual Studio I needed to install the Windows 10 SDK:
        ```

- Install OpenCV: Visit the official OpenCV website, choose the appropriate version for your platform, and download the executable file. Install it on your computer.

- Install CMake: Visit the CMake official website and download the latest release. Install it on your computer.

- Configure Environmental Variables: Add the OpenCV installation directory and the CMake installation directory to your environmental variables.

- Open Visual Studio Code: Create a new project and install the necessary extensions: C/C++ and CMake Tools.

- Create CMakeLists.txt File: Copy and paste the provided code into your CMakeLists.txt file. This code includes commands to find and include the OpenCV directories and link the OpenCV libraries.
  - Add Following to Cmake Dir
    - set(OpenCV_DIR "C:/Dev/opencv/build/x64/vc16/lib")
  - https://stackoverflow.com/questions/70942644/how-to-fix-warning-found-opencv-windows-pack-but-it-has-no-binaries-compatible


- Configure CMake: Access the command palette in Visual Studio Code, select "CMake: Configure," and choose the compiler.

- Build the Project: Click the build button in Visual Studio Code to compile and build the project. Verify that the build process completes successfully.

- Test OpenCV Setup: Use the provided sample code to read and display an image. Build and run the project to confirm that OpenCV is working correctly.

Follow these step-by-step instructions to install OpenCV, set it up in Visual Studio Code using CMake, and verify the OpenCV setup.