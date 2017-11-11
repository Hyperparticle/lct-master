/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <string.h>
#include "debug.h"

void print_heap(struct heap *heap) {
    if (heap->root == NULL) {
        return;
    }

    struct node *next = heap->root;
    struct node *last = next;

    printf("[%d] ", heap->root_list_count);

    do {
        printf("%d (%d), ", next->key, next->degree);
        next = next->right;

    } while (next != last);
    printf("\n");
}

static int check_children(struct heap *heap, struct node *node) {
    if (node == NULL) {
        return 0;
    }

    int count = 0, total = 0;
    struct node *next = node;
    do {
        if (next->child != NULL) {
            int children = check_children(heap, next->child);
            if (children != next->degree) {
                printf("Child count discrepancy\n");
            }
            total += children;
        } else if (next->degree != 0) {
            printf("Child count discrepancy\n");
        }

        count++;
        next = next->right;
    } while (next != node);

    total += count;

    if (total > heap->capacity) {
        printf("Too many nodes\n");
    }

    return count;
}

void check_heap(struct heap *heap) {
    if (heap->root == NULL) {
        return;
    }

    struct node **join_buffer = calloc(heap->max_degree + 1, sizeof(struct node *));
    struct node *next = heap->root;

    int count = 0;
    do {
        if (join_buffer[next->degree] != NULL) {
            print_heap(heap);
            struct node *same = join_buffer[next->degree];
            fprintf(stderr, "Multiple trees of the same order! (%d)\n", next->degree);
        }

        join_buffer[next->degree] = next;
        next = next->right;
        count++;
    } while (next != heap->root);

    if (count != heap->root_list_count) {
        print_heap(heap);
        printf("Root list count discrepancy (%d, %d)\n", heap->root_list_count, count);
    }

    check_children(heap, heap->root);
}