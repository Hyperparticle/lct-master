/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include "tree.h"

struct tree splay_tree = { NULL, 0, 0, NULL, 0 };

struct node *init_node(int key) {
    struct node *node = &splay_tree.node_buffer[splay_tree.node_buffer_i];
    node->left = node->right = node->parent = NULL;
    node->key = key;

    splay_tree.node_buffer_i++;

    return node;
}

void reset_tree(int capacity) {
    if (splay_tree.node_buffer != NULL) {
        free(splay_tree.node_buffer);
    }

    splay_tree.root = NULL;
    splay_tree.tree_size = 0;
    splay_tree.capacity = capacity;

    splay_tree.node_buffer = malloc(capacity * sizeof(struct node));
    splay_tree.node_buffer_i = 0;
}

void insert(int key, bool naive, int *path_length) {
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

    if (naive) {
        splay_naive(insert);
    } else {
        splay(insert);
    }

    splay_tree.root = insert;
    splay_tree.tree_size++;
}

struct node *find(int key, bool naive, int *path_length) {
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
            if (naive) {
                splay_naive(current);
            } else {
                splay(current);
            }

            splay_tree.root = current;

            return current;
        }
    }

    // Couldn't find the node with key
    return NULL;
}
