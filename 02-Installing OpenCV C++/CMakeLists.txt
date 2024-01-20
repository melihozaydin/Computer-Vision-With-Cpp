cmake_minimum_required(VERSION 3.5.0)
project(opencvtest VERSION 0.1.0)

include(CTest)
enable_testing()
# Added to fix error
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/Dev/opencv")
#set(OpenCV_DIR "C:/Dev/opencv/")
set(OpenCV_INCLUDE_DIRS "C:/Dev/opencv/lib")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(opencvtest main.cpp)

target_link_libraries( opencvtest ${OpenCV_LIBS} )

# Ctest to enable testing
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
