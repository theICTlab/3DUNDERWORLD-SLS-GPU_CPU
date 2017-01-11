# For colors
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(ColourBold  "${Esc}[1m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
    set(BoldRed     "${Esc}[1;31m")
    set(BoldGreen   "${Esc}[1;32m")
    set(BoldYellow  "${Esc}[1;33m")
    set(BoldBlue    "${Esc}[1;34m")
    set(BoldMagenta "${Esc}[1;35m")
    set(BoldCyan    "${Esc}[1;36m")
    set(BoldWhite   "${Esc}[1;37m")
endif()

cmake_minimum_required(VERSION 3.1)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )
project( SLS )

# Include this project
include_directories(./src/lib)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package( GLM REQUIRED )
include_directories(${GLM_INCLUDE_DIR})


find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

#Detecting CUDA
find_package(CUDA)
# Set default ENABLE_CUDA value
option (ENABLE_CUDA "Enable CUDA in build" OFF)
option (GTEST "Enable Google test in buid" OFF)
#if (NOT DEFINED ${ENABLE_CUDA})
#    if (${CUDA_FOUND})
#        set(ENABLE_CUDA ON)
#    else (${CUDA_FOUND})
#        set(ENABLE_CUDA OFF)
#    endif(${CUDA_FOUND})
#    message( "CUDA not defined")
#endif (NOT DEFINED ${ENABLE_CUDA})

if (${CUDA_FOUND} AND ${ENABLE_CUDA})
    set( USE_CUDA ON)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    list(APPEND CUDA_NVCC_FLAGS "-O2 -std=c++11 -D__STRICT_ANSI__ -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
else (${CUDA_FOUND} AND ${ENABLE_CUDA})
    set( USE_CUDA OFF)
endif (${CUDA_FOUND} AND ${ENABLE_CUDA})

if (${CUDA_FOUND} AND NOT ${ENABLE_CUDA})
    message(STATUS "${Yellow}CUDA is available but disabled, use -DENABLE_CUDA=ON to enable CUDA${ColourReset}") 
endif (${CUDA_FOUND} AND NOT ${ENABLE_CUDA})

if (NOT ${CUDA_FOUND} AND ${ENABLE_CUDA})
    message(STATUS "${Yellow}CUDA is not found${ColourReset}") 
endif (NOT ${CUDA_FOUND} AND ${ENABLE_CUDA})

    
if (${USE_CUDA})
    add_subdirectory(./src/lib/ReconstructorCUDA)
endif(${USE_CUDA})


set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -O3")

# Enable code coverage
if (COVERAGE)
    set (CMAKE_CXX_FLAGS "-std=c++11 -g -O0 -fprofile-arcs -ftest-coverage")
    set (CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
endif(COVERAGE)


add_subdirectory(./src/lib/GrayCode)
add_subdirectory(./src/lib/calibration)
add_subdirectory(./src/lib/core)
add_subdirectory(./src/app)

if (GTEST)
    enable_testing()
    add_subdirectory(./test)
endif (GTEST)


set_target_properties ( SLS
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

