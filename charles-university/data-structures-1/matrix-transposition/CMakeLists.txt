cmake_minimum_required(VERSION 3.7.2)
project(matrix_transposition)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES_SIM src/cachesim.c)
add_executable(cachesim ${SOURCE_FILES_SIM})

set(SOURCE_FILES src/main.c src/matrix.c src/matrix.h)
add_executable(matrix ${SOURCE_FILES})

target_link_libraries(matrix m)