cmake_minimum_required(VERSION 2.8)
project( SLS )
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -O3")
include_directories(./src/lib)

add_subdirectory(./src/lib/core)
add_subdirectory(./src/app)

