/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#ifndef SPLAY_TREE_TREE_H
#define SPLAY_TREE_TREE_H

#include <stdlib.h>
#include "node.h"

/** The number of nodes in the tree */
long tree_size;

/**
 * Inserts the given key into the tree as a recursive binary tree operation,
 * then splays it to the root.
 */
void insert(int key);

/**
 * Searches the given key as a recursive binary tree operation,
 * splays it to the root (if it exists), and returns the node.
 * @return The node with the given key value if it exists, NULL otherwise.
 */
struct node *find(int key);

#endif //SPLAY_TREE_TREE_H
