/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#ifndef SPLAY_TREE_OPERATION_H
#define SPLAY_TREE_OPERATION_H

/** A splay tree operation code */
enum opcode {
    RESET, INSERT, FIND
};

/**
 * An operation on a splay tree with a given type and value
 * to reset a tree, insert a key, or find a key.
 */
struct operation {
    enum opcode type;
    int value;
};

/**
 * Parses the given string (e.g., "# 1000") as a splay tree operation.
 * @return The operation corresponding to the given string.
 */
struct operation parse_operation(char *str);

/**
 * Performs the given operation
 * @return the path length to find/insert a key (a reset operation returns 0)
 */
int do_operation(struct operation op);

#endif //SPLAY_TREE_OPERATION_H
