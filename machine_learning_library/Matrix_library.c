#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

Matrix* matrix_create(int rows, int cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->data = (double*)malloc(sizeof(double) * rows * cols);
    mat->rows = rows;
    mat->cols = cols;
    return mat;
}

Matrix* matrix_zero_create(int rows, int cols) {
    Matrix* mat = matrix_create(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        mat->data[i] = 0.0;
    return mat;
}

Matrix* matrix_random_create(int rows, int cols) {
    srand(time(NULL)); 
    Matrix* mat = matrix_create(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        mat->data[i] = (double)(rand() / RAND_MAX); 
    return mat;
}

Matrix* matrix_identity_create(int rows) {
    Matrix* mat = matrix_zero_create(rows, rows);
    for (int i = 0; i < rows; i++)
        mat->data[i * rows + i] = 1.0;  
    return mat;
}

void matrix_free(Matrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

double matrix_get(Matrix* mat, int row, int col) {
    return mat->data[row * mat->cols + col]; 
}

void matrix_set(Matrix* mat, int row, int col, double value) {
    mat->data[row * mat->cols + col] = value; 
}

Matrix* matrix_add(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols)
        return NULL;

    Matrix* result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++)
        result->data[i] = a->data[i] + b->data[i];
    return result;
}

Matrix* matrix_sub(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols)
        return NULL;

    Matrix* result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++)
        result->data[i] = a->data[i] - b->data[i];
    return result;
}

Matrix* matrix_scalar_mul(Matrix* a, double scalar) {
    for (int i = 0; i < a->rows * a->cols; i++)
        a->data[i] *= scalar;
    return a;
}

Matrix* matrix_scalar_div(Matrix* a, double scalar) {
    if (scalar == 0) return NULL;  
    for (int i = 0; i < a->rows * a->cols; i++)
        a->data[i] /= scalar;
    return a;
}

Matrix* matrix_mul(Matrix* a, Matrix* b) {
    if (a->cols != b->rows)
        return NULL;

    Matrix* result = matrix_create(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            result->data[i * b->cols + j] = 0;
            for (int k = 0; k < a->cols; k++) {
                result->data[i * b->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }
    return result;
}

Matrix* matrix_transpose(Matrix* mat) {
    Matrix* transpose = matrix_create(mat->cols, mat->rows);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            transpose->data[j * mat->rows + i] = mat->data[i * mat->cols + j]; 
        }
    }
    return transpose;
}
