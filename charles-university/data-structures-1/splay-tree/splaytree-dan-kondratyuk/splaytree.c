/**
 * Analyzes splay tree performance. Given standard input from splaygen, outputs average path lengths for all find operations (calculated per tree) to standard output.
 *
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "operation.h"

/**
 * Reads line by line from the input stream performing the
 * specified operations, and prints path length statistics.
 */
void read_input(FILE *stream);

/**
 * Given the sum of path lengths, the number of paths, and the size of
 * the tree, prints the average path length for the tree.
 */
void print_average(unsigned long path_length_sum, unsigned long path_length_count, int tree_size);

/**
 * Prints how to use the program
 */
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
    const size_t line_size = 200;
    char line[line_size];

    // Used for statistics
    unsigned long path_length_sum = 0, path_length_count = 0;

    int tree_size = 0;
    struct operation op;

    // Read line by line
    // Perform each operation and print statistics
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
            default:
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
