/**
 * Defines simple and recursive matrix transposition algorithms
 * 
 * @date 2017/11/19
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

// The minimum threshold at which the recursive
// algorithm can switch to the simple transpose algorithm
// for a submatrix with n x n elements
#define TRANSPOSE_THRESHOLD (1024) // 1 Ki

/** Recursively transposes the matrix along the diagonal */
static void transpose_diagonal(struct matrix a);

/** Recursively transposes and swaps the matrices with each other */
static void transpose_swap(struct matrix a, struct matrix b);

/** Slices the matrix into 4 parts, modifying the given axx matrices accordingly */
static inline void slice(struct matrix m,
                         struct matrix *a11, struct matrix *a12,
                         struct matrix *a21, struct matrix *a22);

/** Swaps the contents of two integers */
static void swap(int *i, int *j);

/** Returns a pointer to the memory location of the submatrix at i,j */
static inline int *m_data(struct matrix m, int i, int j);

struct matrix matrix_create(unsigned int n) {
    unsigned int size = n * n;
    int *data = malloc(size * sizeof(int));

    struct matrix m = {n, n, n, 0, 0, data};

    return m;
}

void matrix_free(struct matrix m) {
    free(m.data);
}

void transpose_simple(struct matrix m) {
    for (int i = 0; i < m.height; i++) {
        for (int j = i + 1; j < m.width; j++) {
#ifndef PRINT_SWAP // Compiler flag: run an actual matrix transposition or just print all operations
            swap(m_data(m, i, j), m_data(m, j, i));
#else
            printf("X %d %d %d %d\n", m.i + i, m.j + j, m.j + j, m.i + i);
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
        for (int i = 0; i < a.height; i++) {
            for (int j = 0; j < a.width; j++) {
#ifndef PRINT_SWAP // Compiler flag: run an actual matrix transposition or just print all operations
                swap(m_data(a, i, j), m_data(b, j, i));
#else
                printf("X %d %d %d %d\n", a.i + i, a.j + j, a.j + j, a.i + i);
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
    // Slice into 4 quadrants
    struct matrix b[4] = { {
        m.n, m.height / 2, m.width / 2, m.i, m.j, m.data
    }, {
        m.n, m.height / 2, m.width - m.width / 2,
        m.i, m.j + m.width / 2, m.data
    }, {
        m.n, m.height - m.height / 2, m.width / 2,
        m.i + m.height / 2, m.j, m.data
    }, {
        m.n, m.height - m.height / 2, m.width - m.width / 2,
        m.i + m.height / 2, m.j + m.width / 2, m.data
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
    return &m.data[(m.i + i) * m.n + m.j + j];
}
