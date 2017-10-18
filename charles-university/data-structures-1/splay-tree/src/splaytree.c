/**
 * Analyzes splay tree performance. Given a test file, average path lengths are calculated
 * and
 *
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "operation.h"

void read_input(FILE *stream);
void print_average(unsigned long path_length_sum, unsigned long path_length_count, int tree_size);
void print_usage();

static bool naive = false;

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "-n") == 0) {
        naive = true;
    } else if (argc > 2) {
        print_usage();
    }

    read_input(stdin);

    return 0;
}

void read_input(FILE *stream) {
    // Read line by line
    const size_t line_size = 200;
    char line[line_size];

    unsigned long path_length_sum = 0, path_length_count = 0;
    double average = 0;

    int tree_size = 0;
    struct operation op = { RESET, 0 };

    while (fgets(line, line_size, stream) != NULL)  {
        op = parse_operation(line);

        int value = do_operation(op, naive);

        switch (op.type) {
            case RESET:
                print_average(path_length_sum, path_length_count, tree_size);
                path_length_sum = path_length_count = 0;
                tree_size = op.value;
                break;
            case FIND:
                path_length_sum += value;
                path_length_count += 1;
                break;
        }
    }

    print_average(path_length_sum, path_length_count, tree_size);
}

void print_average(unsigned long path_length_sum, unsigned long path_length_count, int tree_size) {
    if (path_length_count != 0) {
        double average = (double) path_length_sum / (double) path_length_count;
        printf("%d,%f\n", tree_size, average);
    }
}

void print_usage() {
    fprintf(stderr, "Usage: splaytree\n");
    fprintf(stderr, "Reads output of splaygen (stdin) and "
            "outputs the average path length of all find operations\n");
    exit(EXIT_FAILURE);
}
