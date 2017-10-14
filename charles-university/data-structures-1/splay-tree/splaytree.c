/**
 * Analyzes splay tree performance. Given a test file, average path lengths are calculated
 * and
 *
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include "operation.h"
#include "tree.h"

void read_file(char *);

int main(int argc, char *argv) {
    read_file("../test/test-10.txt");

    return 0;
}

void read_file(char *filename) {
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read line by line
    const size_t line_size = 100;
    char *line = malloc(line_size);

    while (fgets(line, line_size, file) != NULL)  {
        struct operation op = parse_operation(line);

        if (op.type == RESET && splay_tree.path_lengths != NULL) {
            for (int i = 0; i < splay_tree.path_lengths_i; i++) {
                printf("%d,", splay_tree.path_lengths[i]);
            }
            printf("\n");
        }

        do_operation(op);
    }

    fclose(file);
}
