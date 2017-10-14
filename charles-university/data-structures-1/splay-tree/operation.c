/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "operation.h"

struct operation parse_operation(char *line) {
    struct operation op;

    // Parse the opcode
    char *opcode_str = strtok(line, " ");

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
