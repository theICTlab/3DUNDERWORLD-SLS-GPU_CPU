add_definitions(-DTEST_DATA_PATH=\"${CMAKE_SOURCE_DIR}/data\")
add_definitions(-DTEST_REF_PATH=\"${CMAKE_SOURCE_DIR}/test/data\")

# Grab gtest
include(ExternalProject)
ExternalProject_Add(gtest
    URL https://github.com/google/googletest/archive/release-1.8.0.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    INSTALL_COMMAND ""
    )

ExternalProject_Get_Property(gtest source_dir binary_dir)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

##### Unit test #######
include_directories(${source_dir}/googletest/include)
set(GTEST_LIBRARIES
    ${binary_dir}/googlemock/gtest/libgtest.a
    ${binary_dir}/googlemock/gtest/libgtest_main.a)


add_executable(runCPUTest RunTest.cpp)
add_dependencies(runCPUTest gtest)
target_link_libraries(runCPUTest ${GTEST_LIBRARIES} Threads::Threads)
target_link_libraries(runCPUTest core)

# Use test 
if (COVERAGE)
    include(CodeCoverage)
    if(CMAKE_COMPILER_IS_GNUCXX)
        target_link_libraries(runCPUTest gcov)
    endif()
    setup_target_for_coverage(coverage runCPUTest coverage)
endif(COVERAGE)

add_test( NAME CPU_TEST 
    COMMAND runCPUTest)
