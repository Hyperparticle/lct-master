/**
 * Defines simple and recursive matrix transposition algorithms
 * 
 * @date 2017/11/19
 * @author Dan Kondratyuk
 */

#ifndef MATRIX_TRANSPOSITION_MATRIX_H
#define MATRIX_TRANSPOSITION_MATRIX_H

//#define PRINT_SWAP

/** 
 * Defines an n x n matrix with relative submatrix offsets
 * to aid in the recursive matrix transpose algorithm
 */
struct matrix {
    unsigned int n;             // The size of the entire matrix (n x n)
    unsigned int height, width; // The size of this submatrix (height x width)
    unsigned int i, j;          // The coordinate offset of this submatrix (i, j)
    int *data;                  // The starting point of the entire matrix
};

/** Creates and returns a heap-allocated matrix of size n x n */
struct matrix matrix_create(unsigned int n);

/** Frees a matrix's data */
void matrix_free(struct matrix m);

/** Transposes a matrix with a simple nested loop */
void transpose_simple(struct matrix m);

/** Transposes a matrix with a recursive cache-oblivious algorithm */
void transpose(struct matrix m);

#endif //MATRIX_TRANSPOSITION_MATRIX_H
