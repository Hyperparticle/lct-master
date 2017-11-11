/**
 * @date 11/04/17
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
static void read_input(FILE *stream, bool naive);

/**
 *
 */
void print_average(unsigned long step_sum, unsigned long step_count, int heap_size);

/**
 * Prints how to use the program
 */
static void print_usage();

static bool naive = false;

int main(int argc, char **argv) {
    FILE *stream = stdin;

    if (argc >= 2) {
//        stream = fopen(argv[1], "r");
        stream = popen(argv[1], "r");
    }

    if (argc >= 3 && strcmp(argv[2], "-n") == 0) {
        naive = true;
    } else if (argc >= 3) {
        print_usage();
    }

    read_input(stream, naive);

    return 0;
}

static void read_input(FILE *stream, bool naive) {
    const size_t line_size = 200;
    char line[line_size];

    // Used for statistics
    unsigned long step_sum = 0, step_count = 0;

    int heap_size = 0;
    struct operation op;

    // Read line by line
    // Perform each operation and print statistics
    while (fgets(line, line_size, stream) != NULL)  {
//        printf("\n%s", line);
        op = parse_operation(line);

        int value = do_operation(op, naive);

        switch (op.type) {
            case RESET:
                print_average(step_sum, step_count, heap_size);
                step_sum = step_count = 0;
                heap_size = op.element;
                break;
            case DELETE_MIN:
                step_sum += value;
                step_count += 1;
                break;
            case DECREASE_KEY:
                break;
            default:
                break;
        }
    }

    print_average(step_sum, step_count, heap_size);
}

void print_average(unsigned long step_sum, unsigned long step_count, int heap_size) {
    if (step_count != 0) {
        double average = (double) step_sum / (double) step_count;
        printf("%d,%f\n", heap_size, average);
    }
}

static void print_usage() {
    fprintf(stderr, "Usage: fibheap [filename] [-n]\n");
    fprintf(stderr, "Reads output of heapgen (stdin or from a file)\n");
    fprintf(stderr, "\tfilename (optional) - file path for input\n");
    fprintf(stderr, "\t-n (optional)       - use naive implementation\n");
    exit(EXIT_FAILURE);
}
