
set(CMAKE_PREFIX_PATH /Users/Tsing/tools/googletest)
find_package(gtest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})
add_executable(SLS_test App.cpp)
target_link_libraries(SLS_test ${GTEST_LIBRARIES} core)

