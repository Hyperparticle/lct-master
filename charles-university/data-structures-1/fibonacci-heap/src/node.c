/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <stdlib.h>
#include <stdio.h>
#include "node.h"

void merge_node(struct node *end_node, struct node *new_end) {
    struct node *leftmost = end_node->right;
    end_node->right = new_end;
    leftmost->left = new_end;
    new_end->left = end_node;
    new_end->right = leftmost;
}

void join(struct node *left, struct node *right) {

}
