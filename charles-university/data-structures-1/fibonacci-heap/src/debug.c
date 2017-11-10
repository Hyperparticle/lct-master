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

    printf("[%d] ", heap->node_count);

    do {
        printf("%d (%d), ", next->key, next->child_count);
        next = next->right;

    } while (next != last);
    printf("\n");
}

void check_heap(struct heap *heap) {
    memset(heap->join_buffer, 0, heap->join_buffer_size * sizeof(struct node *));
    struct node *next = heap->root;

    int count = 0;
    do {
        if (heap->join_buffer[next->child_count] != NULL) {
            struct node *i = heap->join_buffer[next->child_count];
            fprintf(stderr, "Multiple trees of the same order! (%d)\n", next->child_count);
        }

        heap->join_buffer[next->child_count] = next;
        next = next->right;
        count++;
    } while (next != heap->root);

    if (count != heap->node_count) {
        printf("Node count discrepancy (%d, %d)\n", heap->node_count, count);
    }
}