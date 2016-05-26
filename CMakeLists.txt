cmake_minimum_required(VERSION 2.8)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )
project( SLS )
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -O3")
include_directories(./src/lib)
#Detecting CUDA
find_package(CUDA)
IF (${CUDA_FOUND})
    add_subdirectory(./src/lib/ReconstructorCUDA)
ENDIF(${CUDA_FOUND})


add_subdirectory(./src/lib/calibration)
add_subdirectory(./src/lib/core)
add_subdirectory(./src/app)

