cmake_minimum_required(VERSION 3.5.1)
project(splay_tree)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES_GEN src/splaygen.c)
add_executable(splaygen ${SOURCE_FILES_GEN})

set(SOURCE_FILES src/splaytree.c src/node.c src/node.h src/operation.c src/operation.h src/tree.c src/tree.h)
add_executable(splaytree ${SOURCE_FILES})