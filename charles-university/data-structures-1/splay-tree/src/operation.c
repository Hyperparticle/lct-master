/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "operation.h"
#include "node.h"
#include "tree.h"

struct operation parse_operation(char *str) {
    struct operation op;

    // Parse the opcode
    char *opcode_str = strtok(str, " ");

    switch (opcode_str[0]) {
        case '#':
            op.type = RESET;
            break;
        case 'I':
            op.type = INSERT;
            break;
        case 'F':
            op.type = FIND;
            break;
        default:
            printf("Failed to parse opcode");
            exit(EXIT_FAILURE);
    }

    // Parse the value
    char *value_str = strtok (NULL, " ");

    op.value = atoi(value_str);

    return op;
}

int do_operation(struct operation op, bool naive) {
    int path_length = 0;

    switch (op.type) {
        case RESET:
            reset_tree(op.value);
            return 0;
        case INSERT:
            insert(op.value, naive, &path_length);
            return path_length;
        case FIND:
            find(op.value, naive, &path_length);
            return path_length;
    }
}
