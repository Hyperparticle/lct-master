cmake_minimum_required(VERSION 3.8)
project(splay_tree)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES splaygen.c)
add_executable(splay_tree ${SOURCE_FILES})