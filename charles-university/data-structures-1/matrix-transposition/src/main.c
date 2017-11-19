/**
 * @date 2017/11/19
 * @author Dan Kondratyuk
 */

#include "matrix.h"

int main() {
    struct matrix m = matrix_create(10);

    matrix_print(m);
    transpose_simple(m);
    matrix_print(m);

    matrix_free(m);

    return 0;
}