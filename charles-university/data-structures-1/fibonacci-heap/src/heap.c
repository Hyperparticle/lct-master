/**
 * Defines all top-level heap operations (reset, insert, extract_min, decrease_key)
 *
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include "heap.h"

// The global Fibonacci heap
static struct heap fib_heap = { NULL, NULL, 0, 0, 0 };

/**
 * Inserts the given node into the heap
 */
static void heap_insert(struct heap *heap, struct node *node);

/**
 * Joins together all nodes of the same degree in the heap's root list
 * @param steps - Adds steps to the number of steps the operation completed
 */
static void heap_consolidate(struct heap *heap, int *steps);

/**
 * Cuts the node from its parent and reinserts it into the heap
 * @param naive - set to false to perform cascading cuts with marked nodes
 */
static void heap_cut(struct heap *heap, struct node *node, bool naive);

void reset(unsigned int capacity) {
    node_free(fib_heap.root);

    fib_heap.root = NULL;
    fib_heap.capacity = capacity;
    fib_heap.root_list_count = 0;
    fib_heap.max_degree = floor_log2(capacity); // Max degree bounded by floor(log2(n))

    fib_heap.node_buffer = calloc(capacity, sizeof(struct node *));
}

void insert(int element, int key) {
    struct node *node = node_init(element, key);
    fib_heap.node_buffer[element] = node;
    heap_insert(&fib_heap, node);
}

int extract_min(int *steps) {
    struct node *min = fib_heap.root;
    struct node *child = min->child;
    int element = min->element;

    *steps += min->degree;
    fib_heap.root_list_count += min->degree - 1;

    // Disconnect the min node from the heap
    if (min->right == min) {
        fib_heap.root = NULL;
    } else {
        fib_heap.root = min->right;
        min->left->right = min->right;
        min->right->left = min->left;
    }

    fib_heap.node_buffer[min->element] = NULL;
    free(min);

    // Remove parent pointers for all children
    if (child != NULL) {
        struct node *next = child;
        do {
            next->parent = NULL;
            next->marked = false;
            next = next->right;
        } while (next != child);
    }

    // Merge the children nodes with the root list
    fib_heap.root = merge_list(fib_heap.root, child);

    heap_consolidate(&fib_heap, steps);

    return element;
}

void decrease_key(int element, int key, bool naive) {
    struct node *node = fib_heap.node_buffer[element];

    if (node == NULL) {
        return;
    }

    // Ignore if new key is greater than current
    if (key > node->key) {
        return;
    }

    node->key = key;

    if (node->parent != NULL && node->parent->key > key) {
        heap_cut(&fib_heap, node, naive);
    } else if (node->key < fib_heap.root->key) {
        // Node is the new minimum
        fib_heap.root = node;
    }
}

static void heap_insert(struct heap *heap, struct node *node) {
    struct node *min = heap->root;

    node->parent = NULL;
    node->marked = false;
    node->left = node->right = node;

    if (min == NULL) {
        heap->root = node;
        node->left = node->right = node;
    } else {
        merge_list(min, node);

        // Update the minimum element
        if (node->key < min->key)  {
            heap->root = node;
        }
    }

    heap->root_list_count++;
}

static void heap_cut(struct heap *heap, struct node *node, bool naive) {
    if (node == NULL || node->parent == NULL) {
        return;
    }

    // Not a valid heap anymore, need to cut
    if (node == node->right) {
        node->parent->child = NULL;
    } else {
        node->left->right = node->right;
        node->right->left = node->left;
        node->parent->child = node->right;
    }

    node->parent->degree--;

    // Cascade cut if not using naive algorithm
    if (!naive) {
        if (node->parent->marked) {
            heap_cut(heap, node->parent, naive);
        } else {
            node->parent->marked = true;
        }
    }

    heap_insert(heap, node);
}

static void heap_consolidate(struct heap *heap, int *steps) {
    if (heap->root == NULL) {
        return;
    }

    // Initialize a join buffer
    struct node **join_buffer = calloc(heap->max_degree + 1, sizeof(struct node *));


    struct node *next = heap->root; // Start at the root
    bool joining; // Termination condition

    // Join all heaps with the same order
    do {
        joining = false;

        int count = heap->root_list_count;
        for (int i = 0; i < count; i++) {
            int order = next->degree;

            while (join_buffer[order] != NULL) {
                struct node *join_with = join_buffer[order];

                if (join_with == next) {
                    break;
                }

                next = node_join(next, join_with); // Join equal orders (degrees) together
                heap->root = next;

                join_buffer[order] = NULL;

                heap->root_list_count--;
                order++;
                *steps += 1;
                joining = true;
            }

            if (order + 1 >= heap->max_degree) {
                heap->max_degree++;
            }

            join_buffer[order] = next;
            next = next->right;
        }
    } while (joining);

    // Update the root by finding the minimum element in log(n) time
    heap->root = find_min(heap->root);

    free(join_buffer);
}
