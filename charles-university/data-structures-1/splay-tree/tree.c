/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include "tree.h"

long tree_size = 0;

void insert(int key) {
    struct node *current = root;
    struct node *parent = NULL;

    // Navigate the binary tree
    while (current != NULL) {
        parent = current;

        if (key <= current->key) {
            current = current->left;
        } else { // key > current->key
            current = current->right;
        }
    }

    // Create a new node and update pointers
    struct node *insert = init_node(key);
    insert->parent = parent;

    if (parent == NULL) {
        root = insert;
    } else if (key <= parent->key) {
        parent->left = insert;
    } else { // key > parent->key
        parent->right = insert;
    }

    splay(insert);

    tree_size++;
}

struct node *find(int key) {
    struct node *current = root;

    // Navigate the binary tree
    while (current != NULL) {
        if (key < current->key) {
            current = current->left;
        } else if (key > current->key) {
            current = current->right;
        } else {
            // Found the node with key
            splay(current);
            return current;
        }
    }

    // Couldn't find the node with key
    return NULL;
}

