cmake_minimum_required(VERSION 3.5.1)
project(matrix_transposition)

set(CMAKE_C_STANDARD 99)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE_FILES_SIM src/cachesim.c)
add_executable(cachesim ${SOURCE_FILES_SIM})

set(SOURCE_FILES src/main.c src/matrix.c src/matrix.h)
add_executable(matrix ${SOURCE_FILES})
add_executable(matrix-print ${SOURCE_FILES})

target_link_libraries(matrix m)
target_link_libraries(matrix-print m)
target_compile_definitions(matrix-print PRIVATE PRINT_SWAP)
