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
    int child_count;
    bool marked;
};

/** Add a node to the end of a circular linked list */
void merge_node(struct node *end_node, struct node *new_end);

void join(struct node *left, struct node *right);

#endif //FIBONACCI_HEAP_NODE_H
