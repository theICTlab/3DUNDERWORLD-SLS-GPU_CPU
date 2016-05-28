aux_source_directory(./ SOURCES)
add_library(sls_graycode ${SOURCES})
target_link_libraries(sls_graycode ${OpenCV_LIBS} )
