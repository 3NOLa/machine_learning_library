#pragma once
#include <time.h>
#include <stdlib.h>

typedef struct {
    int dims; 
    int *shape; //shaoes of the dims
    int *strides; //amount of bytes i need to get to next dim;
    int count; //amount of elemnts
    double* data;
} Tensor;

Tensor* tensor_create(int dims, int* shape);
Tensor* tensor_zero_create(int dims, int* shape);
Tensor* tensor_random_create(int dims, int* shape);
Tensor* tensor_identity_create(int row); 
int tensor_add_row(Tensor* t);

void tensor_free(Tensor* t);
int tensor_copy(Tensor* dest, Tensor* src);
void tensor_zero(Tensor* t);

// Access functions
int tensor_get_index(Tensor* t, int* indices);
double tensor_get_element(Tensor* t, int* indices);
double tensor_get_element_by_index(Tensor* t, int index);
void tensor_set(Tensor* t, int* indices, double value);
void tensor_set_by_index(Tensor* t, int index, double value);

// Dimension manipulation
Tensor* tensor_reshape(Tensor* t, int dims, int* shape);
Tensor* tensor_flatten(Tensor* t); // Convert to 1D tensor
Tensor* tensor_slice_range(Tensor* t, int start, int end);
Tensor* tensor_get_row(Tensor* t, int row);
Tensor* tensor_get_col(Tensor* t, int col);

// Math operations
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_subtract(Tensor* a, Tensor* b);
Tensor* tensor_multiply(Tensor* a, Tensor* b); // Element-wise multiplication
Tensor* tensor_dot(Tensor* a, Tensor* b);      // Matrix multiplication when applicable
Tensor* tensor_add_scalar(Tensor* t, double scalar);
Tensor* tensor_multiply_scalar(Tensor* t, double scalar);
double tensor_sum(Tensor* t);
double tensor_mean(Tensor* t);

// In-place operations (to minimize memory allocations)
void tensor_add_inplace(Tensor* target, Tensor* other);
void tensor_add_more_inplace(Tensor* target, Tensor* others[], int amount);
void tensor_subtract_inplace(Tensor* target, Tensor* other);
void tensor_multiply_inplace(Tensor* target, Tensor* other);
void tensor_add_scalar_inplace(Tensor* target, double scalar);
void tensor_multiply_scalar_inplace(Tensor* target, double scalar);

// Print tensor
void tensor_print(Tensor* t);
