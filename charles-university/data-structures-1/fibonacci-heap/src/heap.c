/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "heap.h"
#include "debug.h"

static struct heap fib_heap = { NULL, NULL, 0, 0, 0 };

static void heap_insert(struct heap *heap, struct node *node);
static void heap_consolidate(struct heap *heap, int *steps);
static void heap_disconnect(struct heap *heap, struct node *node);
static void heap_cut(struct heap *heap, struct node *node, bool naive);

void reset(unsigned int capacity) {
    node_free(fib_heap.root);

    fib_heap.root = NULL;
    fib_heap.capacity = capacity;
    fib_heap.root_list_count = 0;
    fib_heap.max_degree = floor_log2(capacity); // Max degree bounded by floor(log2(n))

    fib_heap.node_buffer = calloc(capacity, sizeof(struct node *));

//    printf("\n# %d\n", capacity);
}

void insert(int element, int key) {
    struct node *node = node_init(element, key);
    fib_heap.node_buffer[element] = node;
    heap_insert(&fib_heap, node);

//    printf("ins: ");
//    print_heap(&fib_heap);
}

void delete_min(int *steps) {
    struct node *min = fib_heap.root;
    struct node *child = min->child;

    *steps += min->degree;
    fib_heap.root_list_count += min->degree - 1;

    heap_disconnect(&fib_heap, min);

    // Remove parent pointers for all children
    if (child != NULL) {
        struct node *next = child;
        do {
            next->parent = NULL;
            next->marked = false;
            next = next->right;
        } while (next != child);
    }


    fib_heap.root = merge_list(fib_heap.root, child);

//    printf("del: ");
//    print_heap(&fib_heap);

    heap_consolidate(&fib_heap, steps);

//    printf("con: ");
//    print_heap(&fib_heap);
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

//    printf("dec: ");
//    print_heap(&fib_heap);
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

static void heap_disconnect(struct heap *heap, struct node *node) {
    if (node->right == node) {
        heap->root = NULL;
    } else {
        heap->root = node->right;
        node->left->right = node->right;
        node->right->left = node->left;
    }

    fib_heap.node_buffer[node->element] = NULL;
    free(node);
}

static void heap_consolidate(struct heap *heap, int *steps) {
    if (heap->root == NULL) {
        return;
    }

    // Initialize a join buffer
    struct node **join_buffer = calloc(heap->max_degree + 1, sizeof(struct node *));

    // Join all heaps with the same order
    struct node *next = heap->root;

    bool joining;
    do {
        joining = false;

        int count = heap->root_list_count;
        for (int i = 0; i < count; i++) {
            int order = next->degree;

            while (join_buffer[order] != NULL) {
                if (order >= heap->max_degree) {
                    fprintf(stderr, "Order greater than join buffer size (%d)\n", order);
                    exit(EXIT_FAILURE);
                }

                struct node *join_with = join_buffer[order];

                if (join_with == next) {
                    break;
                }

                next = join(next, join_with);
                joining = true;
                heap->root = next;
                heap->root_list_count--;
                steps++;

                join_buffer[order] = NULL;
                order++;
            }

            if (order + 1 >= heap->max_degree) {
                heap->max_degree++;
            }

            join_buffer[order] = next;
            next = next->right;
        }
    } while (joining);

    heap->root = find_min(heap->root);

    free(join_buffer);
}
