#pragma once
#include <time.h>
#include <stdlib.h>
#include "export.h"

EXPORT typedef struct {
    int dims; 
    int *shape; //shaoes of the dims
    int *strides; //amount of bytes i need to get to next dim;
    int count; //amount of elemnts
    float * data;
} Tensor;

EXPORT Tensor* tensor_create(int dims, int* shape);
EXPORT Tensor* tensor_zero_create(int dims, int* shape);
EXPORT Tensor* tensor_random_create(int dims, int* shape);
EXPORT Tensor* tensor_identity_create(int row);
EXPORT int tensor_add_row(Tensor* t);

EXPORT void tensor_free(Tensor* t);
EXPORT int tensor_copy(Tensor* dest, Tensor* src);
EXPORT void tensor_zero(Tensor* t);

// Access functions
EXPORT int tensor_get_index(Tensor* t, int* indices);
EXPORT float  tensor_get_element(Tensor* t, int* indices);
EXPORT float  tensor_get_element_by_index(Tensor* t, int index);
EXPORT void tensor_set(Tensor* t, int* indices, float  value);
EXPORT void tensor_set_by_index(Tensor* t, int index, float  value);

// Dimension manipulation
EXPORT Tensor* tensor_reshape(Tensor* t, int dims, int* shape);
EXPORT Tensor* tensor_flatten(Tensor* t); // Convert to 1D tensor
EXPORT Tensor* tensor_slice_range(Tensor* t, int start, int end);
EXPORT Tensor* tensor_get_row(Tensor* t, int row);
EXPORT Tensor* tensor_get_col(Tensor* t, int col);

// Math operations
EXPORT Tensor* tensor_add(Tensor* a, Tensor* b);
EXPORT Tensor* tensor_subtract(Tensor* a, Tensor* b);
EXPORT Tensor* tensor_multiply(Tensor* a, Tensor* b); // Element-wise multiplication
EXPORT Tensor* tensor_dot(Tensor* a, Tensor* b);      // Matrix multiplication when applicable
EXPORT Tensor* tensor_add_scalar(Tensor* t, float  scalar);
EXPORT Tensor* tensor_multiply_scalar(Tensor* t, float  scalar);
EXPORT float  tensor_sum(Tensor* t);
EXPORT float  tensor_mean(Tensor* t);

// In-place operations (to minimize memory allocations)
EXPORT void tensor_add_inplace(Tensor* target, Tensor* other);
EXPORT void tensor_add_more_inplace(Tensor* target, Tensor* others[], int amount);
EXPORT void tensor_subtract_inplace(Tensor* target, Tensor* other);
EXPORT void tensor_multiply_inplace(Tensor* target, Tensor* other);
EXPORT void tensor_add_scalar_inplace(Tensor* target, float  scalar);
EXPORT void tensor_multiply_scalar_inplace(Tensor* target, float  scalar);

// Print tensor
EXPORT void tensor_print(Tensor* t);
