cmake_minimum_required(VERSION 3.5.1)
project(hashing C)

set(CMAKE_C_STANDARD 99)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE_FILES
    src/hash-scheme.c
    src/hash-scheme.h
    src/hash-system.c
    src/hash-system.h
    src/main.c
    src/random-gen.c
    src/random-gen.h)

add_executable(hashing ${SOURCE_FILES})