/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <string.h>
#include "debug.h"

void print_join_buffer(struct heap *heap) {
    if (heap->join_buffer == NULL) {
        return;
    }


    printf("[[%d]] ", heap->join_buffer_size);

    for (int i = 0; i < heap->join_buffer_size; i++) {
        if (heap->join_buffer[i] != NULL) {
            printf("%d (i = %d), ", heap->join_buffer[i]->key, i);
        }
    }

    printf("\n");
}

void print_heap(struct heap *heap) {
    if (heap->root == NULL) {
        return;
    }

    struct node *next = heap->root;
    struct node *last = next;

    printf("[%d] ", heap->root_list_count);

    do {
        printf("%d (%d), ", next->key, next->child_count);
        next = next->right;

    } while (next != last);
    printf("\n");
}

void check_heap(struct heap *heap) {
    if (heap->root == NULL) {
        return;
    }

    memset(heap->join_buffer, 0, heap->join_buffer_size * sizeof(struct node *));
    struct node *next = heap->root;

    int count = 0;
    do {
        if (heap->join_buffer[next->child_count] != NULL) {
            print_heap(heap);
            struct node *same = heap->join_buffer[next->child_count];
            fprintf(stderr, "Multiple trees of the same order! (%d)\n", next->child_count);
        }

        heap->join_buffer[next->child_count] = next;
        next = next->right;
        count++;
    } while (next != heap->root);

    if (count != heap->root_list_count) {
        print_heap(heap);
        printf("Root list count discrepancy (%d, %d)\n", heap->root_list_count, count);
    }
}