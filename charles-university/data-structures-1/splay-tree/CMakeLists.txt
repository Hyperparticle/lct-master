cmake_minimum_required(VERSION 3.8)
project(splay_tree)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES_GEN splaygen.c)
add_executable(splaygen ${SOURCE_FILES_GEN})

set(SOURCE_FILES splaytree.c node.c node.h operation.c operation.h tree.c tree.h)
add_executable(splaytree ${SOURCE_FILES})