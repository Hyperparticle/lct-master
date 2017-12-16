cmake_minimum_required(VERSION 3.7.2)
project(hashing C)

set(CMAKE_C_STANDARD 99)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(hashing src/main.c src/hash-system.c src/hash-system.h src/hash-scheme.c src/hash-scheme.h src/random-gen.c src/random-gen.h)