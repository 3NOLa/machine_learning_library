#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

Matrix* matrix_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "Error: Invalid matrix dimensions (%d x %d)\n", rows, cols);
        return NULL;
    }

    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        fprintf(stderr, "Error: Memory allocation failed for matrix\n");
        return NULL;
    }

    mat->data = (double*)malloc(sizeof(double) * rows * cols);
    if (!mat->data) {
        fprintf(stderr, "Error: Memory allocation failed for matrix data\n");
        free(mat);
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;
    return mat;
}

Matrix* matrix_zero_create(int rows, int cols) {
    Matrix* mat = matrix_create(rows, cols);
    if (!mat) return NULL;

    for (int i = 0; i < rows * cols; i++)
        mat->data[i] = 0.0;
    return mat;
}

Matrix* matrix_random_create(int rows, int cols) {
    static int seed_initialized = 0;
    if (!seed_initialized) {
        srand(time(NULL));
        seed_initialized = 1;
    }

    Matrix* mat = matrix_create(rows, cols);
    if (!mat) return NULL;

    for (int i = 0; i < rows * cols; i++)
        mat->data[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Range [-1, 1]
    return mat;
}

Matrix* matrix_identity_create(int rows) {
    Matrix* mat = matrix_zero_create(rows, rows);
    if (!mat) return NULL;

    for (int i = 0; i < rows; i++)
        mat->data[i * rows + i] = 1.0;
    return mat;
}

void matrix_free(Matrix* mat) {
    if (mat) {
        if (mat->data) free(mat->data);
        free(mat);
    }
}


Matrix* get_row(Matrix* m,int row) {
    if (!m) {
        fprintf(stderr, "Error: NULL matrix get_row\n");
        return NULL;
    }

    Matrix* m_row = (Matrix*)malloc(sizeof(Matrix));
    if (!m_row) {
        fprintf(stderr, "Error: Memory allocation failed for matrix in get_row\n");
        return NULL;
    }
    m_row->cols = m->cols;
    m_row->rows = 1;
    m_row->data = &m->data[row * m->cols];

    return m_row;
}
/*Matrix* get_col(Matrix* m, int col) {
    if (!m) {
        fprintf(stderr, "Error: NULL matrix get_row\n");
        return NULL;
    }

    Matrix* m_row = (Matrix*)malloc(sizeof(Matrix));
    if (!m_row) {
        fprintf(stderr, "Error: Memory allocation failed for matrix in get_row\n");
        return NULL;
    }
    m_row->cols = m->cols;
    m_row = 1;
    m_row->data = &m->data[col * m->rows];

    return m_row;
}*/

double matrix_get(Matrix* mat, int row, int col) {
    if (!mat) {
        fprintf(stderr, "Error: NULL matrix in matrix_get\n");
        return 0.0;
    }
    if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols) {
        fprintf(stderr, "matrix get Error: Index out of bounds (%d,%d) for matrix size (%d,%d)\n",
            row, col, mat->rows, mat->cols);
        return 0.0;
    }
    return mat->data[row * mat->cols + col];
}

void matrix_set(Matrix* mat, int row, int col, double value) {
    if (!mat) {
        fprintf(stderr, "Error: NULL matrix in matrix_set\n");
        return;
    }
    if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols) {
        fprintf(stderr, "matrix set Error: Index out of bounds (%d,%d) for matrix size (%d,%d)\n",
            row, col, mat->rows, mat->cols);
        return;
    }
    mat->data[row * mat->cols + col] = value;
}

int matrix_copy(Matrix* dest, Matrix* src) {
    if (!dest || !src) {
        fprintf(stderr, "Error: NULL matrix in matrix_copy\n");
        return 0;
    }
    if (dest->cols != src->cols || dest->rows != src->rows) {
        fprintf(stderr, "Error: Size mismatch in matrix_copy (%d,%d) vs (%d,%d)\n",
            dest->rows, dest->cols, src->rows, src->cols);
        return 0;
    }

    for (int i = 0; i < src->rows * src->cols; i++)
        dest->data[i] = src->data[i];

    return 1;
}

Matrix* matrix_add(Matrix* a, Matrix* b) {
    if (!a || !b) {
        fprintf(stderr, "Error: NULL matrix in matrix_add\n");
        return NULL;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Size mismatch in matrix_add (%d,%d) vs (%d,%d)\n",
            a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix* result = matrix_create(a->rows, a->cols);
    if (!result) return NULL;

    for (int i = 0; i < a->rows * a->cols; i++)
        result->data[i] = a->data[i] + b->data[i];
    return result;
}

Matrix* matrix_sub(Matrix* a, Matrix* b) {
    if (!a || !b) {
        fprintf(stderr, "Error: NULL matrix in matrix_sub\n");
        return NULL;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Size mismatch in matrix_sub (%d,%d) vs (%d,%d)\n",
            a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix* result = matrix_create(a->rows, a->cols);
    if (!result) return NULL;

    for (int i = 0; i < a->rows * a->cols; i++)
        result->data[i] = a->data[i] - b->data[i];
    return result;
}

Matrix* matrix_scalar_mul(Matrix* a, double scalar) {
    if (!a) {
        fprintf(stderr, "Error: NULL matrix in matrix_scalar_mul\n");
        return NULL;
    }

    Matrix* result = matrix_create(a->rows, a->cols);
    if (!result) return NULL;

    for (int i = 0; i < a->rows * a->cols; i++)
        result->data[i] = a->data[i] * scalar;
    return result;
}

Matrix* matrix_scalar_div(Matrix* a, double scalar) {
    if (!a) {
        fprintf(stderr, "Error: NULL matrix in matrix_scalar_div\n");
        return NULL;
    }
    if (scalar == 0) {
        fprintf(stderr, "Error: Division by zero in matrix_scalar_div\n");
        return NULL;
    }

    Matrix* result = matrix_create(a->rows, a->cols);
    if (!result) return NULL;

    for (int i = 0; i < a->rows * a->cols; i++)
        result->data[i] = a->data[i] / scalar;
    return result;
}

Matrix* matrix_mul(Matrix* a, Matrix* b) {
    if (!a || !b) {
        fprintf(stderr, "Error: NULL matrix in matrix_mul\n");
        return NULL;
    }
    if (a->cols != b->rows) {
        fprintf(stderr, "Error: Incompatible dimensions for matrix_mul (%d,%d) * (%d,%d)\n",
            a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix* result = matrix_create(a->rows, b->cols);
    if (!result) return NULL;

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
    if (!mat) {
        fprintf(stderr, "Error: NULL matrix in matrix_transpose\n");
        return NULL;
    }

    Matrix* transpose = matrix_create(mat->cols, mat->rows);
    if (!transpose) return NULL;

    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            transpose->data[j * mat->rows + i] = mat->data[i * mat->cols + j];
        }
    }
    return transpose;
}