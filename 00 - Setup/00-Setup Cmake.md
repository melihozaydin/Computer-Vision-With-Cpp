# What is CMake

CMake is a cross-platform build system generator for C/C++ projects. It simplifies the process of building, testing, and packaging software across different platforms and compilers.

1. Uses CMakeLists.txt files to define project structure
Generates native build files (e.g., Makefiles, Visual Studio projects)
Manages dependencies and complex build configurations

Use case example:

Cmakefile.txt
```
cmake_minimum_required(VERSION 3.10)
project(MyApp)

add_executable(MyApp main.cpp)
target_link_libraries(MyApp PRIVATE SomeLibrary)
```

Usage:
``` bash
mkdir build && cd build
cmake ..
cmake --build .
```


This example defines a simple project, links a library, and demonstrates the basic workflow of using CMake to generate and execute a build.

---


# Setup Cmake

- Download and install the latest version of CMake from [CMake's official website](https://cmake.org/download/).
- Add CMake to your system's PATH environment variable.
- Verify the installation by opening a terminal or command prompt and typing `cmake --version`.

*Note*: If you are using Visual Studio Code, you can also install the CMake extension from the Visual Studio Code Marketplace.

---
## For linux
For Linux:

1. Install CMake using your package manager:
   - Ubuntu/Debian: `sudo apt-get install cmake`
   - Fedora: `sudo dnf install cmake`
   - Arch Linux: `sudo pacman -S cmake`

2. Verify installation:
   ```
   cmake --version
   ```

3. Optional: Install build essentials (includes GCC, G++, and Make):
   - Ubuntu/Debian: `sudo apt-get install build-essential`

---
# CMake Usage

1. Create a `CMakeLists.txt` file in your project's root directory.

2. Basic structure of a `CMakeLists.txt` file:
   ```cmake
   cmake_minimum_required(VERSION 3.10)
   project(YourProjectName)
   
   add_executable(YourExecutableName source1.cpp source2.cpp)
   ```

3. Create a build directory:
   ```
   mkdir build
   cd build
   ```

4. Generate build files:
   ```
   cmake ..
   ```

5. Build your project:
   ```
   cmake --build .
   ```

6. Run your executable:
   ```
   ./YourExecutableName
   ```

Additional CMake commands:
- `include_directories(directory)`: Add include directories
- `add_library(LibraryName STATIC source1.cpp source2.cpp)`: Create a static library
- `target_link_libraries(YourExecutableName LibraryName)`: Link libraries to your executable

For more complex projects, you can use CMake to:
- Set compiler flags
- Find and use external libraries
- Create installation rules
- Generate config files
- And much more!

Refer to the [CMake documentation](https://cmake.org/documentation/) for detailed information on these features.