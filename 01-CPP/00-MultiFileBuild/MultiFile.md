project/
├── CMakeLists.txt
├── main.cpp
├── math/
│   ├── add.cpp
│   ├── add.h
│   ├── multiply.cpp
│   ├── multiply.h


Building the Project
Navigate to the project directory:

Create a build directory:

Run CMake to generate the build files:

Build the project:

Run the executable:

This setup demonstrates a basic multi-file C++ project using CMake as the build system. 

The CMakeLists.txt file specifies the project name, the source files, and the include directories. 
The main.cpp file includes the headers for the add and multiply functions, which are defined in separate source files.