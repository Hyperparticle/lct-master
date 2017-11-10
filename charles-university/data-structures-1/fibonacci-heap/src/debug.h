/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#ifndef FIBONACCI_HEAP_DEBUG_H
#define FIBONACCI_HEAP_DEBUG_H

#include "heap.h"

void print_join_buffer(struct heap *heap);
void print_heap(struct heap *heap);
void check_heap(struct heap *heap);

#endif //FIBONACCI_HEAP_DEBUG_H
