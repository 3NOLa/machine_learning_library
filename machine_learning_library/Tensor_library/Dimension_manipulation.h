#pragma once
#include "export.h"


typedef struct Tensor Tensor;

EXPORT void transpose_blocked(const float* src, float* dst, int rows, int cols);
EXPORT Tensor* tensor_transpose(Tensor* t);
EXPORT void tensor_transpose_inplace(Tensor* t);

EXPORT Tensor* tensor_reshape(Tensor* t, int dims, int* shape);
EXPORT Tensor* tensor_flatten(Tensor* t); // Convert to 1D tensor
EXPORT Tensor* tensor_slice_range(Tensor* t, int start, int end);
EXPORT Tensor* tensor_slice_dim(Tensor* t, int start, int end,int dim);
EXPORT void tensor_squeeze(Tensor* t);

EXPORT Tensor* tensor_get_row(Tensor* t, int row);
EXPORT bool tensor_add_row(Tensor* t);
