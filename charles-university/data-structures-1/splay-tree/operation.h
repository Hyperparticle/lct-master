/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#ifndef SPLAY_TREE_OPERATION_H
#define SPLAY_TREE_OPERATION_H

enum opcode {
    RESET = 0, INSERT, FIND
};

struct operation {
    enum opcode type;
    int value;
};

struct operation parse_operation(char *);

#endif //SPLAY_TREE_OPERATION_H
