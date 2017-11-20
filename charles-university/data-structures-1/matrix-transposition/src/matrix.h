/**
 * @date 2017/11/19
 * @author Dan Kondratyuk
 */

#ifndef MATRIX_TRANSPOSITION_MATRIX_H
#define MATRIX_TRANSPOSITION_MATRIX_H

// #define PRINT_SWAP

struct matrix {
    unsigned int n, height, width;
    int *data;
};

struct matrix matrix_create(unsigned int n);
void matrix_free(struct matrix m);
void matrix_print(struct matrix m);

void transpose_simple(struct matrix m);
void transpose(struct matrix m);

#endif //MATRIX_TRANSPOSITION_MATRIX_H
