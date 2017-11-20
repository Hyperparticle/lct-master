/**
 * @date 2017/11/19
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "matrix.h"

static struct matrix matrix_k(int k) {
    unsigned int n = (unsigned int) ceil(pow(2, (double) k / 9.0));
//    printf("%d\n", n);
    return matrix_create(n);
}

static bool valid_transpose(struct matrix m) {
    int prev = -1;
    for (int i = 0; i < m.height; i++) {
        for (int j = 0; j < m.width; j++) {
            int next = m.data[j * m.n + i];
            if (prev >= next) {
                return false;
            }

            prev = next;
        }
    }

    return true;
}

int main() {
    clock_t begin, end;

    for (unsigned int k = 54; k < 150; k++) {
        struct matrix m_simple = matrix_k(k);
        struct matrix m = matrix_k(k);

        printf("%d,", m.n);

        begin = clock();
        transpose_simple(m_simple);
        end = clock();
        printf("%f,", (double)(end - begin) / CLOCKS_PER_SEC);

        begin = clock();
        transpose(m);
        end = clock();
        printf("%f\n", (double)(end - begin) / CLOCKS_PER_SEC);

//        transpose_simple(m);
//        if (!valid_transpose(m)) {
//            printf("Fail!\n");
//        }

        matrix_free(m_simple);
        matrix_free(m);
    }

    return 0;
}