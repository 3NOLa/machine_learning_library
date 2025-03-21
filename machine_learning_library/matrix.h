#pragma once
#include <time.h>
#include <stdlib.h>

typedef struct {
    int rows;
    int cols;
    double* data;
} Matrix;

Matrix* matrix_create(int rows, int cols);
Matrix* matrix_zero_create(int rows, int cols);
Matrix* matrix_random_create(int rows, int cols);
Matrix* matrix_identity_create(int rows);
void matrix_free(Matrix* m);

Matrix* get_row(Matrix* m);
Matrix* get_col(Matrix* m);
double matrix_get(Matrix* mat, int row, int col);
void matrix_set(Matrix* mat, int row, int col, double value);
int matrix_copy(Matrix* a, Matrix* b);

Matrix* matrix_add(Matrix* a, Matrix* b);
Matrix* matrix_sub(Matrix* a, Matrix* b);

Matrix* matrix_scalar_mul(Matrix* a, double scalar);
Matrix* matrix_scalar_div(Matrix* a, double scalar);

Matrix* matrix_mul(Matrix* a, Matrix* b);

Matrix* matrix_transpose(Matrix* mat);
