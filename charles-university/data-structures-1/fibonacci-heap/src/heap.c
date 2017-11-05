/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "heap.h"

static struct heap fib_heap = { NULL, 0, NULL, 0, NULL };

static void heap_insert(struct heap *heap, struct node *node);
static void heap_consolidate(struct heap *heap);
static void heap_disconnect(struct heap *heap, struct node *node);

/**
 * Allocates space for a blank node and returns it.
 * @return A node with the given key and pointers set to NULL.
 */
static struct node *init_node(int element, int key);

void reset(int capacity) {
    if (fib_heap.node_buffer != NULL) {
        free(fib_heap.node_buffer);
    }

    if (fib_heap.join_buffer != NULL) {
        free(fib_heap.join_buffer);
    }

    fib_heap.root = NULL;
    fib_heap.capacity = capacity;

    fib_heap.node_buffer = malloc(capacity * sizeof(struct node));
    fib_heap.node_buffer_i = 0;

    fib_heap.join_buffer_size = ceil_log2((unsigned) capacity);
    fib_heap.join_buffer = malloc(fib_heap.join_buffer_size * sizeof(struct node *));
}

void insert(int element, int key, bool naive) {
    struct node *node = init_node(element, key);
    heap_insert(&fib_heap, node);
}

void delete_min(bool naive, int *steps) {
    struct node *min = fib_heap.root;
    struct node *child = min->child;

    heap_disconnect(&fib_heap, min);

    // Remove parent pointers for all children
    if (child != NULL) {
        struct node *next = child->right;
        do {
            next->parent = NULL;
            next->marked = false;
            next = next->right;
        } while (next != child);
    }

    fib_heap.root = merge_list(fib_heap.root, child);

    heap_consolidate(&fib_heap);

    memset(fib_heap.join_buffer, 0, fib_heap.join_buffer_size * sizeof(struct node *));
    struct node *next = fib_heap.root->right;

    do {
        if (fib_heap.join_buffer[next->child_count] != NULL) {
            struct node *i = fib_heap.join_buffer[next->child_count];
            fprintf(stderr, "Multiple trees of the same order! (%d)\n", next->child_count);
            heap_consolidate(&fib_heap);
        }
        fib_heap.join_buffer[next->child_count] = next;
        next = next->right;
    } while (next != fib_heap.root);


}

void decrease_key(int element, int key, bool naive, int *steps) {

}

static struct node *init_node(int element, int key) {
    if (fib_heap.node_buffer_i >= fib_heap.capacity) {
        fprintf(stderr, "Heap capacity exceeded! (%d)\n", fib_heap.capacity);
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

static void heap_insert(struct heap *heap, struct node *node) {
    struct node *min = heap->root;

    node->parent = NULL;
    node->marked = false;
    node->left = node->right = node;

    if (min == NULL) {
        // Initial node
        heap->root = node;
        node->left = node->right = node;
    } else {
        merge_list(min, node);

        if (node->key < min->key)  {
            // Update the minimum element
            heap->root = node;
        }
    }
}

static void heap_disconnect(struct heap *heap, struct node *node) {
    if (node->right == node) {
        heap->root = NULL;
    } else {
        node->left->right = node->right;
        node->right->left = node->left;
        heap->root = node->right;
    }

    node->left = node->right = node->parent = node->child = NULL;
    node->marked = false;
}

static void heap_consolidate(struct heap *heap) {
    if (heap->root == NULL) {
        return;
    }

    // Reset the join buffer
    memset(heap->join_buffer, 0, heap->join_buffer_size * sizeof(struct node *));

    // Join all heaps with the same order
    struct node *next = heap->root;
    struct node *last = next;

    do {
        int order = next->child_count;

        struct node *joined = next;
        heap->root = next = next->right;

        while (heap->join_buffer[order] != NULL) {
            if (order >= heap->join_buffer_size) {
                fprintf(stderr, "Order greater than join buffer size (%d)\n", order);
                exit(EXIT_FAILURE);
            }

            struct node *join_with = heap->join_buffer[order];
//            if (joined == join_with) {
//                break;
//            }

            joined = join(joined, join_with);
            heap->join_buffer[order] = NULL;
            order++;
        }

        heap->join_buffer[order] = joined;
    } while (next != last);

    // Find the minimum
    next = heap->root;
    last = next->left;

    do {
        heap->root = next->key < heap->root->key ? next : heap->root;
        next = next->right;
    } while (next != last);
}
