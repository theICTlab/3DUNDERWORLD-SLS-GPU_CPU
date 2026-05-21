set(ABSL_PROPAGATE_CXX_STD ON)

aux_source_directory(. SOURCES)
add_executable(SLS App.cpp)
target_link_libraries(SLS core)


IF (${USE_CUDA})
    cuda_add_executable(SLS_GPU ./App_CUDA.cu)
    target_link_libraries(SLS_GPU core sls_gpu)
    set_target_properties(SLS_GPU
            PROPERTIES
            #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
            #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
ENDIF (${USE_CUDA})

add_executable(calibrateCamera CalibrateCamera.cpp)
target_link_libraries(calibrateCamera core sls_calib)

add_executable(generateGraycode generateGraycode.cpp)
target_link_libraries(generateGraycode core sls_graycode)

add_executable(sync_main sync_main.cc)
target_link_libraries(sync_main core core absl::base
        sls_graycode
        absl::strings
        absl::status absl::statusor absl::flags
        absl::flags_parse
        absl::log
        absl::log_initialize
        absl::check)

set_target_properties(SLS
        PROPERTIES
        #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(calibrateCamera
        PROPERTIES
        #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(generateGraycode
        PROPERTIES
        #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(sync_main
        PROPERTIES
        #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(FetchContent)

# Abseil
FetchContent_Declare(
        abseil
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
        GIT_TAG 20260107.1
)
FetchContent_MakeAvailable(abseil)

