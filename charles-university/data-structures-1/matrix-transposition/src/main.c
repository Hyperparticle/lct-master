/**
 * @date 2017/11/19
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "matrix.h"

// #define START_K 54
// #define END_K 150
#define START_K 55
#define END_K 56

static void benchmark();
static void print_benchmark();

static inline unsigned int n_val(unsigned int k) {
    return (unsigned int) ceil(pow(2, (double) k / 9.0));
}

// static bool valid_transpose(struct matrix m) {
//     int prev = -1;
//     for (int i = 0; i < m.height; i++) {
//         for (int j = 0; j < m.width; j++) {
//             int next = m.data[j * m.n + i];
//             if (prev >= next) {
//                 return false;
//             }

//             prev = next;
//         }
//     }

//     return true;
// }

int main(int argc, char **argv) {
#ifndef PRINT_SWAP
    benchmark();
#else
    print_benchmark();
#endif

    return 0;
}

static void benchmark() {
    clock_t begin, end;

    for (unsigned int k = START_K; k < END_K; k++) {
        struct matrix m_simple = matrix_create(n_val(k));
        struct matrix m = matrix_create(n_val(k));

        begin = clock();
        transpose_simple(m_simple);
        end = clock();
        printf("%f,", (double)(end - begin) / CLOCKS_PER_SEC);

        begin = clock();
        matrix_print(m);
        transpose(m);
        matrix_print(m);
        end = clock();
        printf("%f\n", (double)(end - begin) / CLOCKS_PER_SEC);

//        transpose_simple(m);
//        if (!valid_transpose(m)) {
//            printf("Fail!\n");
//        }

        matrix_free(m_simple);
        matrix_free(m);
    }
}

static void print_benchmark() {
    for (unsigned int k = START_K; k < END_K; k++) {
        unsigned int n = n_val(k);
        struct matrix m = { n, n, n, NULL };

        printf("N %d\n", m.n);
        transpose_simple(m);
        printf("E\n");
        
        matrix_free(m);
    }
}
