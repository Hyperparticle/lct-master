/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#ifndef SPLAY_TREE_NODE_H
#define SPLAY_TREE_NODE_H

#include <stdbool.h>

/** A splay tree node. Contains a key value along with pointers to its left, right, and parent subtrees. */
struct node {
    struct node *left, *right, *parent;
    int key;
};

/** @return whether the given node is the root node */
bool is_root(struct node *node);

/** Given a parent node, its right node is rotated left to become the new parent. */
void rotate_left(struct node *p);

/** Given a parent node, its left node is rotated right to become the new parent. */
void rotate_right(struct node *p);

/** Given a node x whose parent is the root, x is rotated to be the new root. */
void zig(struct node *x);

/**
 * Given a node x whose parent p and grandparent g are both left or right
 * children, p is rotated up to g and then x is rotated up to p.
 */
void zig_zig(struct node *x);

/**
 * Given a node x whose parent p and grandparent g are neither both left or
 * right children, x is rotated up to p and then rotated up to g.
 */
void zig_zag(struct node *x);

/**
 * A splay operation that moves the given node up to the root using the
 * defined conditions for the zig, zig-zig, and zig-zag operations.
 */
void splay(struct node *x);

/** A splay operation that moves the given node up to the root using only simple rotations. */
void splay_naive(struct node *x);

#endif //SPLAY_TREE_NODE_H
