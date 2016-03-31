aux_source_directory(. SOURCES)
add_executable(SLS App.cpp)
target_link_libraries( SLS core)
