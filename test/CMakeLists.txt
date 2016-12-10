
# Grab gtest
include(ExternalProject)
ExternalProject_Add(gtest
  URL https://github.com/google/googletest/archive/release-1.8.0.zip
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
  INSTALL_COMMAND ""
)

ExternalProject_Get_Property(gtest source_dir binary_dir)
message(STATUS "SOURCE DIR" ${surce_dir})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

##### Unit test #######
include_directories(${source_dir}/googletest/include)
set(GTEST_LIBRARIES
    ${binary_dir}/googlemock/gtest/libgtest.a
    ${binary_dir}/googlemock/gtest/libgtest_main.a)


add_executable(runCPUTest RunTest.cpp splitstring.cpp)
add_dependencies(runCPUTest gtest)
target_link_libraries(runCPUTest ${GTEST_LIBRARIES} Threads::Threads)
target_link_libraries(runCPUTest core)

add_test( NAME CPU_TEST 
    COMMAND runCPUTest)

