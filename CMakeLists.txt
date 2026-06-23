# For colors
if (NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(ColourBold "${Esc}[1m")
    set(Red "${Esc}[31m")
    set(Green "${Esc}[32m")
    set(Yellow "${Esc}[33m")
    set(Blue "${Esc}[34m")
    set(Magenta "${Esc}[35m")
    set(Cyan "${Esc}[36m")
    set(White "${Esc}[37m")
    set(BoldRed "${Esc}[1;31m")
    set(BoldGreen "${Esc}[1;32m")
    set(BoldYellow "${Esc}[1;33m")
    set(BoldBlue "${Esc}[1;34m")
    set(BoldMagenta "${Esc}[1;35m")
    set(BoldCyan "${Esc}[1;36m")
    set(BoldWhite "${Esc}[1;37m")
endif ()

cmake_minimum_required(VERSION 3.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
project(SLS)
set(VERSION_MAJOR 4)
set(VERSION_MINOR 0)
set(VERSION_PATCH 1)


# Include this project
include_directories(./src/lib)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIR})

add_definitions(-DGLM_ENABLE_EXPERIMENTAL)
find_package(OpenCV 4.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# On Windows the prebuilt OpenCV uses include/opencv2/ but sources include
# <opencv4/opencv2/...> (the Linux /usr/include/opencv4 layout). Provide
# a compat dir where opencv4/ maps to the actual include root.
if(WIN32 AND DEFINED OPENCV4_COMPAT_DIR)
    include_directories(${OPENCV4_COMPAT_DIR})
endif()

# GPU backend selection: NVIDIA via ENABLE_CUDA (legacy FindCUDA), AMD via
# USE_HIP (first-class HIP language). USE_GPU gates the shared GPU targets.
option (ENABLE_CUDA "Enable CUDA in build" OFF)
option (USE_HIP "Build the GPU reconstructor with HIP for AMD GPUs" OFF)
option (GTEST "Enable Google test in buid" OFF)
set( USE_CUDA OFF)
set( USE_GPU OFF)

if (USE_HIP)
    # enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, else auto-detects the
    # host GPU, else errors on a no-GPU host.
    enable_language(HIP)
    set( USE_GPU ON)
    add_definitions(-DUSE_HIP)
    message(STATUS "${Green}Building GPU reconstructor with HIP for ${CMAKE_HIP_ARCHITECTURES}${ColourReset}")
else (USE_HIP)
    #Detecting CUDA
    find_package(CUDA)
    if (${CUDA_FOUND} AND ${ENABLE_CUDA})
        set( USE_CUDA ON)
        set( USE_GPU ON)
        include_directories(${CUDA_INCLUDE_DIRS})
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        list(APPEND CUDA_NVCC_FLAGS "-O2 -std=c++11 -D__STRICT_ANSI__ -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
    endif (${CUDA_FOUND} AND ${ENABLE_CUDA})

    if (${CUDA_FOUND} AND NOT ${ENABLE_CUDA})
        message(STATUS "${Yellow}CUDA is available but disabled, use -DENABLE_CUDA=ON to enable CUDA${ColourReset}")
    endif (${CUDA_FOUND} AND NOT ${ENABLE_CUDA})

    if (NOT ${CUDA_FOUND} AND ${ENABLE_CUDA})
        message(STATUS "${Yellow}CUDA is not found${ColourReset}")
    endif (NOT ${CUDA_FOUND} AND ${ENABLE_CUDA})
endif (USE_HIP)


if (${USE_GPU})
    add_subdirectory(./src/lib/ReconstructorCUDA)
endif(${USE_GPU})


set(CMAKE_CXX_FLAGS "-std=c++20 -g -Wall")

if (GTEST)
    enable_testing()
    add_subdirectory(./test)
endif (GTEST)

# Enable code coverage
option(COVERAGE "Build with coverage test, require GTEST=on" OFF)
if (COVERAGE AND GTEST)
    set(CMAKE_CXX_FLAGS "-std=c++20 -g -O0 -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS="-fprofile-arcs -ftest-coverage")
endif ()

# Enable documentation
option(BUILD_DOC "Build documentation" OFF)
if (BUILD_DOC)
    add_subdirectory("./doc")
endif (BUILD_DOC)


add_subdirectory(./src/lib/GrayCode)
add_subdirectory(./src/lib/calibration)
add_subdirectory(./src/lib/core)
add_subdirectory(./src/app)


set_target_properties(SLS
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

