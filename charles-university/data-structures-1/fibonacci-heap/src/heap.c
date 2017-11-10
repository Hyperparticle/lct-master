/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "heap.h"
#include "debug.h"

static struct heap fib_heap = { NULL, 0, 0, NULL, NULL, 0 };

static void heap_insert(struct heap *heap, struct node *node);
static void heap_consolidate(struct heap *heap, int *steps);
static void heap_disconnect(struct heap *heap, struct node *node);

void reset(int capacity) {
    node_free(fib_heap.root);
    fib_heap.root = NULL;

    if (fib_heap.join_buffer != NULL) {
        free(fib_heap.join_buffer);
    }
    fib_heap.capacity = capacity;

    fib_heap.root_list_count = 0;

    fib_heap.node_buffer = calloc((size_t) capacity, sizeof(struct node *));

    fib_heap.join_buffer_size = ceil_log2((unsigned) capacity);
//    fib_heap.join_buffer_size = capacity;
    fib_heap.join_buffer = calloc((size_t) fib_heap.join_buffer_size, sizeof(struct node *));
}

void insert(int element, int key) {
    struct node *node = node_init(element, key);
    fib_heap.node_buffer[element] = node;
    heap_insert(&fib_heap, node);

//    printf("##");
//    print_heap(&fib_heap);
}

void delete_min(int *steps) {
    struct node *min = fib_heap.root;
    struct node *child = min->child;

    *steps += min->child_count;
    fib_heap.root_list_count += min->child_count - 1;

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

    heap_consolidate(&fib_heap, steps);

//    printf("**");
//    print_heap(&fib_heap);
}

static void cascade_cut(struct node *node) {
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

    node->parent->child_count--;

    if (node->parent->marked) {
        // Cascading cut
        cascade_cut(node->parent);
    } else {
        node->parent->marked = true;
    }

    heap_insert(&fib_heap, node);
}

static void naive_cut(struct node *node) {
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

    node->parent->child_count--;

    heap_insert(&fib_heap, node);
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
        if (naive) {
            naive_cut(node);
        } else {
            cascade_cut(node);
        }


    } else if (node->key < fib_heap.root->key) {
        // Node is the new minimum
        fib_heap.root = node;
    }

//    printf("--");
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
//    print_heap(heap);
    if (heap->root == NULL) {
        return;
    }

    // Reset the join buffer
    memset(heap->join_buffer, 0, heap->join_buffer_size * sizeof(struct node *));

    // Join all heaps with the same order
    struct node *next = heap->root;

    bool joining;
    do {
        joining = false;

        int count = heap->root_list_count;
        for (int i = 0; i < count; i++) {
            int order = next->child_count;

            while (heap->join_buffer[order] != NULL) {
                if (order >= heap->join_buffer_size) {
//                    print_heap(heap);
                    fprintf(stderr, "Order greater than join buffer size (%d)\n", order);
                    exit(EXIT_FAILURE);
                }

                struct node *join_with = heap->join_buffer[order];

                if (join_with == next) {
                    break;
                }

                next = join(next, join_with);
                joining = true;
                heap->root = next;
                heap->root_list_count--;
//                print_heap(heap);
                steps++;

                heap->join_buffer[order] = NULL;
                order++;
            }

            heap->join_buffer[order] = next;
            next = next->right;
        }
    } while (joining);

    heap->root = find_min(heap->root);

//    check_heap(&fib_heap);
}
