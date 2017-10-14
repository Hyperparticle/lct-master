/**
 * @date 10/14/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include "operation.h"

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
    const size_t line_size = 300;
    char* line = malloc(line_size);

    while (fgets(line, line_size, file) != NULL)  {
        struct operation op = parse_operation(line);

        printf("%d\n", op.value);
    }

    fclose(file);
}
