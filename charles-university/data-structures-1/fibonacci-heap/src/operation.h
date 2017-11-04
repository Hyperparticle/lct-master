/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#ifndef FIBONACCI_HEAP_OPERATION_H
#define FIBONACCI_HEAP_OPERATION_H

#include <stdbool.h>

/** A Fibonacci heap operation code */
enum opcode {
    RESET, INSERT, DELETE_MIN, DECREASE_KEY
};

/**
 * An operation on a Fibonacci heap with a given type and values
 */
struct operation {
    enum opcode type;
    int element;
    int key;
};

/**
 * Parses the given string (e.g., "# 1000") as a Fibonacci heap operation.
 * @return The operation corresponding to the given string.
 */
struct operation parse_operation(char *str);

/**
 * Performs the given operation
 * @return the number of steps executed for this operation, i.e,
 * the number of children of the deleted node that are appended to
 * the list of trees, plus the number of times some tree is
 * linked to another tree during the subsequent consolidation
 * procedure.
 * A reset operation returns 0.
 */
int do_operation(struct operation op, bool naive);

#endif //FIBONACCI_HEAP_OPERATION_H
