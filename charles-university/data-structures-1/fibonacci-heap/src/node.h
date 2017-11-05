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

/** Merge two circular linked lists together */
struct node *merge_list(struct node *list0, struct node *list1);

struct node *join(struct node *left, struct node *right);

/**
 * Computes the ceiling of the log base 2 of the input
 * @see https://stackoverflow.com/a/15327567/6485996
 */
int ceil_log2(unsigned long long x);

#endif //FIBONACCI_HEAP_NODE_H
