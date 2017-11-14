/**
 * Defines low-level node operations on heap nodes (init, free, merge, join, find_min)
 *
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#ifndef FIBONACCI_HEAP_NODE_H
#define FIBONACCI_HEAP_NODE_H

#include <stdbool.h>

/**
 * A Fibonacci heap node. The left and right pointers point to its
 * left and right siblings in a circular doubly-linked list. As such,
 * only one child pointer is necessary to traverse the list.
 */
struct node {
    struct node *left, *right, *parent, *child; // Pointers to sibling, parent, and child nodes
    int element;         // A unique element identifier
    int key;             // The priority of this node
    unsigned int degree; // The number of children this node has
    bool marked;         // Whether this node has lost at least one child
};

/**
 * Allocates space for a blank node and returns it.
 * @return A node with the given key and pointers set to NULL.
 */
struct node *node_init(int element, int key);

/** Recursively frees the node, its siblings, and its children */
void node_free(struct node *node);

/** Merge two circular linked lists together */
struct node *merge_list(struct node *list0, struct node *list1);

/** Joins two nodes together. The node with the smaller key becomes the parent of the other. */
struct node *node_join(struct node *left, struct node *right);

/** Finds the minimum element in the node list */
struct node *find_min(struct node *list);

/**
 * Computes the floor of the log base 2 of the input
 * @see Bit Twiddling Hacks: http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
 */
unsigned int floor_log2(unsigned int x);

#endif //FIBONACCI_HEAP_NODE_H
