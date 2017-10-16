/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include "tree.h"

//  Uncomment this to run the naive implementation
//#define NAIVE

struct tree splay_tree = { NULL, 0, 0, NULL, 0 };

struct node *init_node(int key) {
    if (splay_tree.node_buffer_i >= splay_tree.capacity) {
        printf("Unable to allocate more nodes: overflow (%d)\n", splay_tree.capacity);
        exit(EXIT_FAILURE);
    }

    struct node *node = (struct node *) malloc(sizeof(struct node));
    node->left = node->right = node->parent = NULL;
    node->key = key;

    splay_tree.node_buffer[splay_tree.node_buffer_i] = node;
    splay_tree.node_buffer_i++;

    return node;
}

void reset_tree(int capacity) {
    if (splay_tree.node_buffer != NULL) {
        for (int i = 0; i < splay_tree.node_buffer_i; i++) {
            free(splay_tree.node_buffer[splay_tree.node_buffer_i]);
        }

        free(splay_tree.node_buffer);
    }

    splay_tree.root = NULL;
    splay_tree.tree_size = 0;
    splay_tree.capacity = capacity;

    splay_tree.node_buffer = malloc(capacity * sizeof(struct node));
    splay_tree.node_buffer_i = 0;
}

void insert(int key, int *path_length) {
    struct node *current = splay_tree.root;
    struct node *parent = NULL;

    *path_length = 0;

    // Navigate the binary tree
    while (current != NULL) {
        parent = current;

        if (key < current->key) {
            current = current->left;
            *path_length += 1;
        } else if (key > current->key) {
            current = current->right;
            *path_length += 1;
        } else {
            *path_length = 0;
            return; // If the key exists, ignore insert
        }
    }

    // Create a new node and update pointers
    struct node *insert = init_node(key);
    insert->parent = parent;

    if (parent == NULL) {
        splay_tree.root = insert;
    } else if (key <= parent->key) {
        parent->left = insert;
    } else { // key > parent->key
        parent->right = insert;
    }

#ifdef NAIVE
    splay_naive(insert);
#else
    splay(insert);
#endif

    splay_tree.root = insert;
    splay_tree.tree_size++;
}

struct node *find(int key, int *path_length) {
    struct node *current = splay_tree.root;

    *path_length = 0;

    // Navigate the binary tree
    while (current != NULL) {
        if (key < current->key) {
            current = current->left;
            *path_length += 1;
        } else if (key > current->key) {
            current = current->right;
            *path_length += 1;
        } else {
            // Found the node with key
#ifdef NAIVE
            splay_naive(current);
#else
            splay(current);
#endif
            splay_tree.root = current;

            return current;
        }
    }

    // Couldn't find the node with key
    return NULL;
}
