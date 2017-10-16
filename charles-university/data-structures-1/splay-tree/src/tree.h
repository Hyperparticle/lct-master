/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#ifndef SPLAY_TREE_TREE_H
#define SPLAY_TREE_TREE_H

#include <stdlib.h>
#include "node.h"

struct tree {
    struct node *root; // The root of the tree
    int tree_size; // The number of nodes in the tree
    int capacity;  // The maximum number of nodes in the tree
    struct node **node_buffer; // A buffer containing all nodes
    int node_buffer_i;
} splay_tree;

/**
 * Allocates space for a blank node and returns it.
 * @return A node with the given key and pointers set to NULL.
 */
struct node *init_node(int key);

/**
 * Resets the tree, freeing any memory from the old tree.
 */
void reset_tree(int capacity);

/**
 * Inserts the given key into the tree as a recursive binary tree operation,
 * then splays it to the root.
 */
void insert(int key, int *path_length);

/**
 * Searches the given key as a recursive binary tree operation,
 * splays it to the root (if it exists), and returns the node.
 * @return The node with the given key value if it exists, NULL otherwise.
 */
struct node *find(int key, int *path_length);

#endif //SPLAY_TREE_TREE_H
