cmake_minimum_required(VERSION 3.9)
project(hashing C)

set(CMAKE_C_STANDARD 99)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(hashing main.c hash-system.c hash-system.h hash-scheme.c hash-scheme.h random-gen.c random-gen.h)