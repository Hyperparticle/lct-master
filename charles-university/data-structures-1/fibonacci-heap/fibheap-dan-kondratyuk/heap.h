/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#ifndef FIBONACCI_HEAP_HEAP_H
#define FIBONACCI_HEAP_HEAP_H

#include <stdlib.h>
#include <stdbool.h>
#include "node.h"

struct heap {
    struct node **node_buffer;    // An array of all nodes in the heap
    struct node *root;            // The root of the heap (minimum element)
    unsigned int capacity;        // The maximum number of nodes in the heap
    unsigned int root_list_count; // The number of nodes in the root of the heap
    unsigned int max_degree;      // The current maximum number of children a node in the root list has
};

/** Resets the heap, freeing any memory from the old heap */
void reset(unsigned int capacity);

/**  Inserts the given key and element */
void insert(int element, int key);

void delete_min(int *steps);

void decrease_key(int element, int key, bool naive);

#endif //FIBONACCI_HEAP_HEAP_H
