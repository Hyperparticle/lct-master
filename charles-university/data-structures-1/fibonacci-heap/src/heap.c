/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include "heap.h"

static struct heap fib_heap = { NULL, 0, NULL, 0 };

static void heap_insert(struct heap *heap, struct node *node);
static void heap_consolidate(struct heap *heap);
void heap_disconnect(struct heap *heap, struct node *node);

struct node *init_node(int element, int key) {
    if (fib_heap.node_buffer_i >= fib_heap.capacity) {
        fprintf(stderr, "Heap capacity exceeded!");
        exit(EXIT_FAILURE);
    }

    struct node *node = &fib_heap.node_buffer[fib_heap.node_buffer_i];
    node->left = node->right = node->parent = node->child = NULL;
    node->child = 0;
    node->marked = false;
    node->element = element;
    node->key = key;

    fib_heap.node_buffer_i++;

    return node;
}

void reset(int capacity) {
    if (fib_heap.node_buffer != NULL) {
        free(fib_heap.node_buffer);
    }

    fib_heap.root = NULL;
    fib_heap.capacity = capacity;

    fib_heap.node_buffer = malloc(capacity * sizeof(struct node));
    fib_heap.node_buffer_i = 0;
}

void insert(int element, int key, bool naive) {
    struct node *node = init_node(element, key);
    heap_insert(&fib_heap, node);
}

static void heap_insert(struct heap *heap, struct node *node) {
    struct node *min = heap->root;

    node->parent = NULL;
    node->marked = false;

    if (min == NULL) {
        // Initial node
        heap->root = node;
        node->left = node->right = node;
    } else {
        merge_node(min, node);

        if (node->key < min->key)  {
            // Update the minimum element
            heap->root = node;
        }
    }
}

void delete_min(bool naive, int *steps) {
    struct node *min = fib_heap.root;
    struct node *child = min->child;
    heap_disconnect(&fib_heap, min);

    // Set children to point parent to root
    if (child != NULL) {
        child->parent = fib_heap.root;
        struct node *next = child->right;

        while (next != child) {
            next->parent = fib_heap.root;
            next = next->right;
        }
    }

    // Merge the two lists












    heap_consolidate(&fib_heap);
}

void heap_disconnect(struct heap *heap, struct node *node) {
    if (node->right == node) {
        heap->root = NULL;
    } else {
        node->left->right = node->right;
        node->right->left = node->left;
        heap->root = node->right;
    }

    node->left = node->right = node->parent = node->child = NULL;
}

void decrease_key(int element, int key, bool naive, int *steps) {

}
