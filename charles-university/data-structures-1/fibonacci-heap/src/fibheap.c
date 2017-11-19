/**
 * Main starting point that reads input from stdin
 *
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
 * specified operations, and prints statistics.
 */
static void read_input(FILE *stream, bool naive);

/**
 * Prints the average number of steps an ExtractMin operation takes given the
 * total number of steps and the sum of all steps.
 */
static void print_average(unsigned long step_sum, unsigned long step_count, int heap_size);

/**
 * Prints how to use the program
 */
static void print_usage();

int main(int argc, char **argv) {
    FILE *stream = stdin;

    bool naive = false;

    stream = popen(argv[1], "r");

//    if (argc >= 2 && strcmp(argv[1], "-n") == 0) {
//        naive = true;
//    } else if (argc >= 3) {
//        print_usage();
//    }

    read_input(stream, naive);

    return 0;
}

static void read_input(FILE *stream, bool naive) {
    const size_t line_size = 200;
    char line[line_size];

    // Used for statistics
    unsigned long step_sum = 0, step_count = 0;

    int heap_size = 0;

    // Read line by line
    // Perform each operation and print statistics
    while (fgets(line, line_size, stream) != NULL)  {
        struct operation op = parse_operation(line);

        int steps = do_operation(op, naive);

        switch (op.type) {
            case RESET:
                print_average(step_sum, step_count, heap_size);
                step_sum = step_count = 0;
                heap_size = op.element;
                break;
            case DELETE_MIN:
                step_sum += steps;
                step_count++;
                break;
            case DECREASE_KEY:
                break;
            default:
                break;
        }
    }

    print_average(step_sum, step_count, heap_size);
}

static void print_average(unsigned long step_sum, unsigned long step_count, int heap_size) {
    if (step_count != 0) {
        double average = (double) step_sum / (double) step_count;
        printf("%d,%lu,%lu\n", heap_size, step_sum, step_count);
    }
}

static void print_usage() {
    fprintf(stderr, "Usage: fibheap [-n]\n");
    fprintf(stderr, "Reads output of heapgen from stdin\n");
    fprintf(stderr, "\t[-n] (optional) - use naive implementation\n");
    exit(EXIT_FAILURE);
}
