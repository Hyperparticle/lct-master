cmake_minimum_required(VERSION 3.7.2)
project(fibonacci_heap)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES_GEN src/heapgen.cpp)
add_executable(heapgen ${SOURCE_FILES_GEN})

set(SOURCE_FILES src/fibheap.c src/node.c src/node.h src/operation.c src/operation.h src/heap.c src/heap.h)
add_executable(fibheap ${SOURCE_FILES})