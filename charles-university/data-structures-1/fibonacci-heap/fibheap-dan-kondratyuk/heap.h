/**
 * Defines all top-level heap operations (reset, insert, extract_min, decrease_key)
 *
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#ifndef FIBONACCI_HEAP_HEAP_H
#define FIBONACCI_HEAP_HEAP_H

#include <stdlib.h>
#include <stdbool.h>
#include "node.h"

/**
 * A Fibonacci heap. Maintains metadata information to help
 * with its various operations.
 */
struct heap {
    struct node **node_buffer;    // An array of all nodes in the heap
    struct node *root;            // The root of the heap (minimum element)
    unsigned int capacity;        // The maximum number of nodes in the heap
    unsigned int root_list_count; // The number of nodes in the root list of the heap
    unsigned int max_degree;      // The current maximum number of children a node in the root list has
};

/** Resets the heap with the new capacity, freeing any memory from the old heap */
void reset(unsigned int capacity);

/** Inserts the given key and element to the heap */
void insert(int element, int key);

/**
 * Extracts the minimum element and consolidates the heap.
 * @param steps - Adds to steps the number of steps the operation completed
 * @return the minimum element extracted
 */
int extract_min(int *steps);

/**
 * Decreases the element's key to the specified value
 * @param naive - whether the operation uses the naive algorithm (i.e., no cascading cuts)
 */
void decrease_key(int element, int key, bool naive);

#endif //FIBONACCI_HEAP_HEAP_H
