aux_source_directory(./ SOURCES)
add_library(calib ${SOURCES})
target_link_libraries(calib ${OpenCV_LIBS})
