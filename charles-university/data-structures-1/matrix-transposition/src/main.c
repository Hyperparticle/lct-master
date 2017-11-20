/**
 * @date 2017/11/19
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

static bool simple = false;

static void print_usage();
static void benchmark(unsigned int k, bool simple);
static void print_benchmark(unsigned int k, bool simple);

static inline unsigned int n_val(unsigned int k) {
    return (unsigned int) ceil(pow(2, (double) k / 9.0));
}

int main(int argc, char **argv) {
    unsigned int k = 54;

    if (argc >= 2 && argc <= 3) {
        k = (unsigned) atoi(argv[1]);
    } else {
        print_usage();
    }

    if (argc == 3) {
        if (strcmp(argv[2], "-s") == 0) {
            simple = true;
        } else {
            print_usage();
        }
    }

#ifndef PRINT_SWAP
    benchmark(k, simple);
#else
    print_benchmark(k, simple);
#endif

    return 0;
}

static void print_usage() {
    fprintf(stderr, "Usage: matrix <k> [-s]\n");
    fprintf(stderr, "k  - k value (n = ceil(2^(k/9)))\n");
    fprintf(stderr, "-s - (optional) run the simple transpose algorithm \n");
    exit(EXIT_FAILURE);
}

static void benchmark(unsigned int k, bool simple) {
    clock_t begin, end;

    struct matrix m = matrix_create(n_val(k));

    begin = clock();
    if (simple) {
        transpose_simple(m);
    } else {
        transpose(m);
    }
    end = clock();

    printf("%d,%f\n", m.n, (double)(end - begin) / CLOCKS_PER_SEC);

    matrix_free(m);
}

static void print_benchmark(unsigned int k, bool simple) {
    unsigned int n = n_val(k);
    struct matrix m = { n, n, n, 0, 0, NULL };

    printf("N %d\n", m.n);
    if (simple) {
        transpose_simple(m);
    } else {
        transpose(m);
    }
    printf("E\n");

    matrix_free(m);
}
