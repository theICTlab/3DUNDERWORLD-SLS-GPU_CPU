
enable_testing()
set(CMAKE_PREFIX_PATH /Users/Tsing/tools/googletest)
find_package(gtest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})
add_executable(SLS_test RunTest.cpp splitstring.cpp)
target_link_libraries(SLS_test ${GTEST_LIBRARIES} core)
add_test( SLS_test SLS_test)

