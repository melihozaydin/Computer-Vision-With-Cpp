Example of a multi-file C++ project using an external library.
We'll use the Boost library for this example.
Specifically, we'll use the Boost Filesystem library to demonstrate how to include and link against an external library using CMake.

Directory Structure
project/
├── CMakeLists.txt
├── main.cpp
├── math/
│   ├── add.cpp
│   ├── add.h
│   ├── multiply.cpp
│   ├── multiply.h

Building the Project
Install Boost:
    sudo apt-get install libboost-all-dev

Navigate to the project directory:

    cd ./project

Create a build directory:
    mkdir build
    cd build

Run CMake to generate the build files:
    cmake ..

Build the project:
    make

Run the executable:
    ./MyProject
