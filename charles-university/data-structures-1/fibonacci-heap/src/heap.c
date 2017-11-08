/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "heap.h"

static struct heap fib_heap = { NULL, 0, 0, 0, NULL, 0, NULL };

static void heap_insert(struct heap *heap, struct node *node);
static void heap_consolidate(struct heap *heap, int *steps);
static void heap_disconnect(struct heap *heap, struct node *node);

//static void print_dbg() {
//    if (fib_heap.root == NULL) {
//        return;
//    }
//
//    struct node *next = fib_heap.root;
//    struct node *last = next;
//
//    printf("[%d] ", fib_heap.node_count);
//
//    do {
//        printf("%d (%d), ", next->key, next->child_count);
//        next = next->right;
//    } while (next != last);
//    printf("\n");
//}

//static void check_heap() {
//    memset(fib_heap.join_buffer, 0, fib_heap.join_buffer_size * sizeof(struct node *));
//    struct node *next = fib_heap.root;
//
//    do {
//        if (fib_heap.join_buffer[next->child_count] != NULL) {
//            struct node *i = fib_heap.join_buffer[next->child_count];
//            fprintf(stderr, "Multiple trees of the same order! (%d)\n", next->child_count);
//        }
//
//        fib_heap.join_buffer[next->child_count] = next;
//        next = next->right;
//    } while (next != fib_heap.root);
//}

void reset(int capacity) {
    node_free(fib_heap.root);
    fib_heap.root = NULL;

    if (fib_heap.join_buffer != NULL) {
        free(fib_heap.join_buffer);
    }
    fib_heap.capacity = capacity;

    fib_heap.node_count = 0;
    fib_heap.join_buffer_size = ceil_log2((unsigned) capacity);
    fib_heap.join_buffer = malloc(fib_heap.join_buffer_size * sizeof(struct node *));
}

void insert(int element, int key) {
    struct node *node = node_init(element, key);
    heap_insert(&fib_heap, node);
}

void delete_min(int *steps) {
    struct node *min = fib_heap.root;
    struct node *child = min->child;

    *steps += min->child_count;
    fib_heap.node_count += min->child_count - 1;

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
}

void decrease_key(int element, int key, bool naive) {

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

    heap->node_count++;
}

static void heap_disconnect(struct heap *heap, struct node *node) {
    if (node->right == node) {
        heap->root = NULL;
    } else {
        node->left->right = node->right;
        node->right->left = node->left;
        heap->root = node->right;
    }

    free(node);
//    node->left = node->right = node->parent = node->child = NULL;
//    node->marked = false;
}

static void heap_consolidate(struct heap *heap, int *steps) {
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

        int count = heap->node_count;
        for (int i = 0; i < count; i++) {
            int order = next->child_count;

            while (heap->join_buffer[order] != NULL) {
                if (order >= heap->join_buffer_size) {
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
                heap->node_count--;
                steps++;

                heap->join_buffer[order] = NULL;
                order++;
            }

            heap->join_buffer[order] = next;

            next = next->right;
        }
    } while (joining);

    // Find the minimum
    next = heap->root;
    struct node *last = next;

    do {
        heap->root = next->key < heap->root->key ? next : heap->root;
        next = next->right;
    } while (next != last);
}
