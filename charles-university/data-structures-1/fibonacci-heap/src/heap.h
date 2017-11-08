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
    struct node *root;        // The root of the heap (minimum element)
    int capacity;             // The maximum number of nodes in the heap
    int r_capacity;
    int node_count;           // The number of nodes in the root of the heap
    struct node **node_buffer; // A buffer containing all nodes
    int node_buffer_i;        // The number of nodes in the heap
    struct node **join_buffer; // Used to join heaps together in the consolidation phase
    int join_buffer_size;
};

/** Resets the heap, freeing any memory from the old heap */
void reset(int capacity);

/**  Inserts the given key and element */
void insert(int element, int key);

void delete_min(int *steps);

void decrease_key(int element, int key, bool naive);

#endif //FIBONACCI_HEAP_HEAP_H
