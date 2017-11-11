/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#ifndef FIBONACCI_HEAP_NODE_H
#define FIBONACCI_HEAP_NODE_H

#include <stdbool.h>

/** A Fibonacci heap node. */
struct node {
    struct node *left, *right, *parent, *child;
    int element;
    int key;
    unsigned int degree;
    bool marked;
};

/**
 * Allocates space for a blank node and returns it.
 * @return A node with the given key and pointers set to NULL.
 */
struct node *node_init(int element, int key);

void node_free(struct node *node);

/** Merge two circular linked lists together */
struct node *merge_list(struct node *list0, struct node *list1);

struct node *join(struct node *left, struct node *right);

struct node *find_min(struct node *node);

/**
 * Computes the floor of the log base 2 of the input
 * @see Bit Twiddling Hacks: http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
 */
unsigned int floor_log2(unsigned int x);

#endif //FIBONACCI_HEAP_NODE_H
