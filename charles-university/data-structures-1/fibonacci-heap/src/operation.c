/**
 * @date 11/04/17
 * @author Dan Kondratyuk
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "operation.h"
#include "node.h"
#include "heap.h"

struct operation parse_operation(char *str) {
    struct operation op = { RESET, 0, 0 };

    // Parse the opcode
    char *opcode_str = strtok(str, " \n");

    // Parse the value
    if (strcmp(opcode_str, "#") == 0) {
        op.type = RESET;

        char *element_str = strtok(NULL, " ");

        op.element = atoi(element_str);
        op.key = 0;
    } else if (strcmp(opcode_str, "INS") == 0) {
        op.type = INSERT;

        char *element_str = strtok(NULL, " ");
        op.element = atoi(element_str);

        char *key_str = strtok(NULL, " ");
        op.key = atoi(key_str);
    } else if (strcmp(opcode_str, "DEL") == 0) {
        op.type = DELETE_MIN;

        op.element = op.key = 0;
    } else if (strcmp(opcode_str, "DEC") == 0) {
        op.type = DECREASE_KEY;

        char *element_str = strtok(NULL, " ");
        op.element = atoi(element_str);

        char *key_str = strtok(NULL, " ");
        op.key = atoi(key_str);
    } else {
        printf("Failed to parse opcode (%s)", opcode_str);
        exit(EXIT_FAILURE);
    }

    return op;
}

int do_operation(struct operation op, bool naive) {
    int steps = 0;

    switch (op.type) {
        case RESET:
            reset(op.element);
            break;
        case INSERT:
            insert(op.element, op.key, naive);
            break;
        case DELETE_MIN:
            delete_min(naive, &steps);
            break;
        case DECREASE_KEY:
            decrease_key(op.element, op.key, naive);
            break;
        default:
            break;
    }

    return steps;
}
