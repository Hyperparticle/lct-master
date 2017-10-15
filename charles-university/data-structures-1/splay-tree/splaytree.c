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

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: splaytree [filename]\n");
        fprintf(stderr, "filename - output of splaygen");
        exit(EXIT_FAILURE);
    }

    read_file(argv[1]);

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
            printf("%d,", splay_tree.capacity);
            printf("%d,", splay_tree.path_lengths_i);

            long sum = 0;

            for (int i = 0; i < splay_tree.path_lengths_i - 1; i++) {
                sum += splay_tree.path_lengths[i];
//                printf("%d,", splay_tree.path_lengths[i]);
            }
//            printf("%d\n", splay_tree.path_lengths[splay_tree.path_lengths_i - 1]);

            double average = (double) sum / (double) splay_tree.path_lengths_i;

            printf("%f\n", average);
        }

        do_operation(op);
    }

    fclose(file);
}
