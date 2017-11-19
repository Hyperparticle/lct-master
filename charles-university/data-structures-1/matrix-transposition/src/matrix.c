/**
 * @date 2017/11/19
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

#define TRANSPOSE_THRESHOLD 4

static void swap(int *i, int *j);
static inline int *m_data(struct matrix m, int i, int j);
static void transpose_diagonal(struct matrix a);
static void transpose_swap(struct matrix a, struct matrix b);

struct matrix matrix_create(unsigned int n) {
    unsigned int size = n * n;
    int *data = malloc(size * sizeof(int));

    struct matrix m = { n, data, 0, 0 };

    for (int i = 0; i < size; i++) {
        data[i] = i;
    }

    return m;
}

void matrix_free(struct matrix m) {
    free(m.data);
}

void matrix_print(struct matrix m) {
    int padding = (int) ceil(log10(m.n * m.n)) + 1;

    for (int i = m.i; i < m.n; i++) {
        for (int j = m.j; j < m.n; j++) {
            printf("%*d", padding, *m_data(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

void transpose_simple(struct matrix m) {
    for (int i = m.i; i < m.n; i++) {
        for (int j = m.j; j < i; j++) {
            swap(m_data(m, i, j), m_data(m, j, i));
        }
    }
}

void transpose(struct matrix m) {
    transpose_diagonal(m);
}

static void transpose_diagonal(struct matrix a) {
    if (a.n <= TRANSPOSE_THRESHOLD) {
        transpose_simple(a);
        return;
    }

    struct matrix a11, a12, a21, a22;

    transpose_diagonal(a11);
    transpose_diagonal(a22);
    transpose_swap(a12, a21);
}

static void transpose_swap(struct matrix a, struct matrix b) {
    if (a.n < TRANSPOSE_THRESHOLD && b.n < TRANSPOSE_THRESHOLD) {
        // Swap chunks of memory
        int tmp[a.n * a.n];
        for (int i = 0; i < a.n; i++) {
            memcpy(tmp, m_data(a, i, 0), a.n);
        }
        return;
    }

    struct matrix a11, a12, a21, a22;
    struct matrix b11, b12, b21, b22;

    transpose_swap(a11, b11);
    transpose_swap(a12, b21);
    transpose_swap(a21, b12);
    transpose_swap(a22, b22);
}

static void swap(int *i, int *j) {
    int tmp = *i;
    *i = *j;
    *j = tmp;
}

static inline int *m_data(struct matrix m, int i, int j) {
    return &m.data[m.i * m.n + m.j + i * m.n + j];
}
