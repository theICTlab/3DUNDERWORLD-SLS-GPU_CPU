aux_source_directory(./ SOURCES)

add_library(core ${SOURCES})
target_link_libraries(core ${OpenCV_LIBS})
