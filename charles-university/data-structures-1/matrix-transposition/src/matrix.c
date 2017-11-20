/**
 * @date 2017/11/19
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

#define TRANSPOSE_THRESHOLD 4 * 1024 // 4 KiB

static void transpose_diagonal(struct matrix a);
static void transpose_swap(struct matrix a, struct matrix b);
static inline void slice(struct matrix m, 
                         struct matrix *a11, struct matrix *a12, 
                         struct matrix *a21, struct matrix *a22);
static void swap(int *i, int *j);
static inline int *m_data(struct matrix m, int i, int j);

struct matrix matrix_create(unsigned int n) {
    unsigned int size = n * n;
    int *data = malloc(size * sizeof(int));

    struct matrix m = { n, n, n, data };

   for (long i = 0; i < size; i++) {
       data[i] = (int) i;
   }

    return m;
}

void matrix_free(struct matrix m) {
    free(m.data);
}

void matrix_print(struct matrix m) {
    int padding = (int) ceil(log10(m.n * m.n)) + 1;

    for (int i = 0; i < m.height; i++) {
        for (int j = 0; j < m.width; j++) {
            printf("%*d", padding, *m_data(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

void transpose_simple(struct matrix m) {
    for (int i = 0; i < m.height; i++) {
        for (int j = 0; j < i; j++) {
#ifndef PRINT_SWAP
            swap(m_data(m, i, j), m_data(m, j, i));
#else
            printf("X %d %d %d %d\n", i, j, j, i);
#endif
        }
    }
}

void transpose(struct matrix m) {
    transpose_diagonal(m);
}

static void transpose_diagonal(struct matrix a) {
    if (a.height * a.width <= TRANSPOSE_THRESHOLD) {
        transpose_simple(a);
        return;
    }

    struct matrix a11, a12, a21, a22;
    slice(a, &a11, &a12, &a21, &a22);

    transpose_diagonal(a11);
    transpose_diagonal(a22);
    transpose_swap(a12, a21);
}

static void transpose_swap(struct matrix a, struct matrix b) {
    if (a.height * a.width <= TRANSPOSE_THRESHOLD &&
        b.height * b.width <= TRANSPOSE_THRESHOLD) {
        // Swap chunks of memory
        for (int i = 0; i < a.height; i++) {
            for (int j = 0; j < a.width; j++) {
#ifndef PRINT_SWAP
                swap(m_data(a, i, j), m_data(b, j, i));
#else
                printf("X %d %d %d %d\n", i, j, j, i);
#endif
                
            }
        }

        return;
    }

    struct matrix a11, a12, a21, a22, b11, b12, b21, b22;
    slice(a, &a11, &a12, &a21, &a22);
    slice(b, &b11, &b12, &b21, &b22);

    transpose_swap(a11, b11);
    transpose_swap(a12, b21);
    transpose_swap(a21, b12);
    transpose_swap(a22, b22);
}

static inline void slice(struct matrix m,
                         struct matrix *a11, struct matrix *a12,
                         struct matrix *a21, struct matrix *a22) {
    struct matrix b[4] = { {
            m.n, m.height / 2, m.width / 2, m.data
    }, {
        m.n, m.height / 2, m.width - m.width / 2,
        m_data(m, 0, m.width / 2)
    }, {
        m.n, m.height - m.height / 2, m.width / 2,
        m_data(m, m.height / 2, 0)
    }, {
        m.n, m.height - m.height / 2, m.width - m.width / 2,
        m_data(m, m.height / 2, m.width / 2)
    } };

    *a11 = b[0];
    *a12 = b[1];
    *a21 = b[2];
    *a22 = b[3];
}

static void swap(int *i, int *j) {
    int tmp = *i;
    *i = *j;
    *j = tmp;
}

static inline int *m_data(struct matrix m, int i, int j) {
    return &m.data[i * m.n + j];
}
