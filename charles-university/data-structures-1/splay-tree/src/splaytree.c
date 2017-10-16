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

void read_input();

int main(int argc, char **argv) {
    if (argc >= 2) {
        fprintf(stderr, "Usage: splaytree\n");
        fprintf(stderr, "Reads output of splaygen (stdin) and "
                "outputs the average path length of all find operations\n");
        exit(EXIT_FAILURE);
    }

    read_input();

    return 0;
}

void read_input() {

    // Read line by line
    const size_t line_size = 100;
    char line[line_size];

    unsigned long path_length_sum = 0;
    unsigned long path_length_count = 0;

    while (fgets(line, line_size, stdin) != NULL)  {
        struct operation op = parse_operation(line);

        int value = do_operation(op);

        if (op.type == FIND) {
            path_length_sum += value;
            path_length_count += 1;
        }

//        printf("%lu\n", path_length_sum);
    }

    double average = (double) path_length_sum / (double) path_length_count;

    printf("%f\n", average);
}
