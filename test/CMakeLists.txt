
if(DEFINED TEST_DATA_PATH AND DEFINED TEST_REF_PATH)
    add_definitions(-DTEST_DATA_PATH=\"${TEST_DATA_PATH}\")
    add_definitions(-DTEST_REF_PATH=\"${TEST_REF_PATH}\")

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

    # Use test 
    if (COVERAGE)
        include(CodeCoverage)
        setup_target_for_coverage(${PROJECT_NAME}_coverage runCPUTest coverage)
    endif(COVERAGE)

    add_test( NAME CPU_TEST 
        COMMAND runCPUTest)

else()
    message( FATAL_ERROR "${Red}GTEST is enabled but TEST_DATA_PATH and TEST_REF_PATH are not set ${ColourReset} \
    ${Yellow}Use -DTEST_DATA_PATH and -DTEST_REF_PATH to set test data folder and reference data folder${ColourReset}")
    message( STATUS "${Yellow}Use -DTEST_DATA_PATH and -DTEST_REF_PATH to set test input folder and reference data folder${ColourReset}")
endif()
