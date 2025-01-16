cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MyProject)

# Find the Boost library
find_package(Boost 1.65 REQUIRED COMPONENTS filesystem)

# Add the executable
add_executable(MyProject main.cpp math/add.cpp math/multiply.cpp)

# Include directories
target_include_directories(MyProject PRIVATE math ${Boost_INCLUDE_DIRS})

# Link against the Boost libraries
target_link_libraries(MyProject PRIVATE ${Boost_LIBRARIES})