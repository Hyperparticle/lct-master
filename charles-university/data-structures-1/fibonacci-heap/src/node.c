/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "node.h"

struct node *node_init(int element, int key) {
    struct node *node = malloc(sizeof(struct node));
    node->left = node->right = node->parent = node->child = NULL;
    node->degree = 0;
    node->marked = false;

    node->element = element;
    node->key = key;

    return node;
}

void node_free(struct node *node) {
    if (node == NULL) {
        return;
    }

    struct node *next = node;
    next->left->right = NULL; // Unravel the linked list for termination

    do {
        struct node *right = next->right;
        node_free(next->child);
        next = right;
    } while (next != NULL);

    free(node);
}

struct node *merge_list(struct node *list0, struct node *list1) {
    if (list0 == NULL) { return list1; }
    if (list1 == NULL) { return list0; }

    struct node *last0 = list0->right;
    struct node *last1 = list1->left;

    list0->right = list1;
    list1->left = list0;
    last0->left = last1;
    last1->right = last0;

    return list0;
}

struct node *join(struct node *left, struct node *right) {
    if (left == right) {
        fprintf(stderr, "Cannot join node with itself\n");
        exit(EXIT_FAILURE);
    }

    if (right->key < left->key) {
        return join(right, left);
    }

    // Detach right
    right->left->right = right->right;
    right->right->left = right->left;
    right->left = right->right = right;

    if (left->child != NULL) {
        merge_list(left->child, right);
    }

    right->parent = left;
    left->child = right;
    left->degree++;

    return left;
}

struct node *find_min(struct node *min) {
    struct node *next = min;
    struct node *last = next;

    do {
        min = next->key < min->key ? next : min;
        next = next->right;
    } while (next != last);

    return min;
}

unsigned int floor_log2(unsigned int v) {
    static const int MultiplyDeBruijnBitPosition[32] = {
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
        8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
    };

    v |= v >> 1; // first round down to one less than a power of 2
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;

    return MultiplyDeBruijnBitPosition[(v * 0x07C4ACDDU) >> 27];
}
