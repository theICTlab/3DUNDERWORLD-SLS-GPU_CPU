aux_source_directory(./ SOURCES)
add_library(sls_calib ${SOURCES})
target_link_libraries(sls_calib ${OpenCV_LIBS})
