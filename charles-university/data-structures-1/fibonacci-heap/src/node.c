/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include "node.h"

struct node *node_init(int element, int key) {
    struct node *node = malloc(sizeof(struct node));
    node->left = node->right = node->parent = node->child = NULL;
    node->child_count = 0;
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
    left->child_count++;

    return left;
}

int ceil_log2(unsigned long long x) {
    static const unsigned long long t[6] = {
            0xFFFFFFFF00000000ull,
            0x00000000FFFF0000ull,
            0x000000000000FF00ull,
            0x00000000000000F0ull,
            0x000000000000000Cull,
            0x0000000000000002ull
    };

    int y = (((x & (x - 1)) == 0) ? 0 : 1);
    int j = 32;
    int i;

    for (i = 0; i < 6; i++) {
        int k = (((x & t[i]) == 0) ? 0 : j);
        y += k;
        x >>= k;
        j >>= 1;
    }

    return y;
}
