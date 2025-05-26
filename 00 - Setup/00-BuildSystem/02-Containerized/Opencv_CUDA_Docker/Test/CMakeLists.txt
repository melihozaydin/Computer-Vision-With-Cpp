cmake_minimum_required(VERSION 3.5.0)
project(opencvGPUtest VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )

# To inspect variable
include(CMakePrintHelpers)
cmake_print_variables(OpenCV_INCLUDE_DIRS)
cmake_print_variables(OpenCV_LIBS)
cmake_print_variables(OpenCV_DIR)
#set(OpenCV_DIR /path/to/opencv_install_dir/lib/cmake/opencv4)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(opencvGPUtest opencvGPUtest.cpp)

target_link_libraries( opencvGPUtest ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
