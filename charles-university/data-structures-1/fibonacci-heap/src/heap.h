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
    struct node *node_buffer; // A buffer containing all nodes
    int node_buffer_i;        // The number of nodes in the heap
};

/**
 * Allocates space for a blank node and returns it.
 * @return A node with the given key and pointers set to NULL.
 */
struct node *init_node(int element, int key);

/**
 * Resets the heap, freeing any memory from the old heap.
 */
void reset(int capacity);

/**  Inserts the given key and element */
void insert(int element, int key, bool naive);

void delete_min(bool naive, int *steps);

void decrease_key(int element, int key, bool naive, int *steps);


#endif //FIBONACCI_HEAP_HEAP_H
