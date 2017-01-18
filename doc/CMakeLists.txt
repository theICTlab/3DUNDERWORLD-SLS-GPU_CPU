find_package(Doxygen REQUIRED)

set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/doxygen.in)
set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doxyfile)
set(doxy_main_page ${CMAKE_SOURCE_DIR}/README.md)

configure_file(${doxyfile_in} ${doxyfile} @ONLY)

add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

