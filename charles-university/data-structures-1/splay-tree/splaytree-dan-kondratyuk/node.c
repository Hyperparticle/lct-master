/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include <stdlib.h>
#include <stdio.h>
#include "node.h"

#include "tree.h"

bool is_root(struct node *node) {
    return node->parent == NULL;
}

void rotate_right(struct node *p) {
    struct node *g = p->parent;
    struct node *x = p->left;

    if (x != NULL) {
        p->left = x->right;

        if (x->right != NULL) {
            x->right->parent = p;
        }

        x->parent = g;
        x->right = p;
    }

    if (g != NULL) {
        if (p == g->left) {
            g->left = x;
        } else { // x == g->right
            g->right = x;
        }
    }

    p->parent = x;
}

void rotate_left(struct node *p) {
    struct node *g = p->parent;
    struct node *x = p->right;

    if (x != NULL) {
        p->right = x->left;
        x->parent = g;

        if (x->left != NULL) {
            x->left->parent = p;
        }

        x->left = p;
    }

    if (g != NULL) {
        if (p == g->left) {
            g->left = x;
        } else { // x == g->right
            g->right = x;
        }
    }

    p->parent = x;
}

void zig(struct node *x) {
    struct node *p = x->parent;

    if (x == p->left) {
        rotate_right(p);
    } else if (x == p->right) {
        rotate_left(p);
    } else {
        printf("There was an error in zig\n");
        exit(EXIT_FAILURE);
    }
}

void zig_zig(struct node *x) {
    struct node *p = x->parent;
    struct node *g = p->parent;

    bool both_left  = p == g->left  && x == p->left;
    bool both_right = p == g->right && x == p->right;

    if (both_left) {
        rotate_right(g);
        rotate_right(p);
    } else if (both_right) {
        rotate_left(g);
        rotate_left(p);
    } else {
        printf("There was an error in zig_zig\n");
        exit(EXIT_FAILURE);
    }
}

void zig_zag(struct node *x) {
    struct node *p = x->parent;
    struct node *g = p->parent;

    bool right_left = p == g->right && x == p->left;
    bool left_right = p == g->left  && x == p->right;

    if (right_left) {
        rotate_right(p);
        rotate_left(g);
    } else if (left_right) {
        rotate_left(p);
        rotate_right(g);
    } else {
        printf("There was an error in zig_zag\n");
        exit(EXIT_FAILURE);
    }
}

void splay(struct node *x) {

    while (!is_root(x)) {
        if (is_root(x->parent)) {
            zig(x);
        } else {
            struct node *p = x->parent;
            struct node *g = p->parent;

            bool both_left  = p == g->left  && x == p->left;
            bool both_right = p == g->right && x == p->right;

            if (both_left || both_right) {
                zig_zig(x);
            } else {
                zig_zag(x);
            }
        }
    }
}

void splay_naive(struct node *x) {
    while (!is_root(x)) {
        struct node *p = x->parent;

        if (x == p->left) {
            rotate_right(p);
        } else { // x == p->right
            rotate_left(p);
        }
    }
}
