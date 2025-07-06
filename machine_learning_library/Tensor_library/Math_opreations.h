#pragma once
#include "export.h"


typedef struct Tensor Tensor;
typedef struct Function Function;

// Math operations

//add
EXPORT void array_add(float *a, float *b, float *result, int count);
EXPORT void add_backward(Function* f, Tensor* output_grad);
EXPORT Tensor* tensor_add(Tensor* a, Tensor* b);
EXPORT void array_add_scalar(float *a, float  scalar, float *result, int count);
EXPORT Tensor* tensor_add_scalar(Tensor* t, float  scalar);
EXPORT void tensor_add_scalar_inplace(Tensor* t, float  scalar);
EXPORT void tensor_add_scalar_inplace(Tensor* target, float  scalar);
EXPORT void tensor_add_inplace(Tensor* target, Tensor* other);
EXPORT void tensor_add_more_inplace(Tensor* target, Tensor* others[], int amount);

//subtract
EXPORT void array_subtract(float *a, float *b, float * result, int count);
EXPORT void array_sub_scalar(float *a, float  scalar, float *result, int count);
EXPORT Tensor* tensor_subtract(Tensor* a, Tensor* b);
EXPORT void subtract_backward(Function* f, Tensor* output_grad);
EXPORT Tensor* tensor_subtract_scalar(Tensor* t, float  scalar);
EXPORT void tensor_subtract_inplace(Tensor* target, Tensor* other);

//multiply
EXPORT void array_multiply(float *a, float *b, float * result, int count); // Element-wise multiplication
EXPORT void array_mul_scalar(float *a, float  scalar, float *result, int count);
EXPORT Tensor* tensor_multiply(Tensor* a, Tensor* b); // Element-wise multiplication
EXPORT void mul_backward(Function* f, Tensor* output_grad);
EXPORT void tensor_mul_scalar_inplace(Tensor* t, float scalar);
EXPORT void tensor_multiply_inplace(Tensor* target, Tensor* other);
EXPORT Tensor* tensor_multiply_scalar(Tensor* t, float  scalar);
EXPORT void tensor_multiply_scalar_exsting(Tensor* dest,Tensor* source, float  scalar);
EXPORT void tensor_multiply_scalar_existing_more(Tensor* dests[], Tensor* sources[], const float scalar[], int amount);

//div
EXPORT void array_div(float *a, float *b, float * result, int count); // Element-wise multiplication
EXPORT void array_div_scalar(float *a, float  scalar, float *result, int count);
EXPORT Tensor* tensor_div(Tensor* a, Tensor* b);
EXPORT Tensor* tensor_div_scalar(Tensor* t, float  scalar);
EXPORT void div_backward(Function* f, Tensor* output_grad); 
EXPORT void tensor_div_scalar_inplace(Tensor* t, float  scalar);

//matrix multipication
EXPORT void matmul(float* a, float* b, float* result, int a_dim, int b_dim, int same_dim);// assuming b is transpoed for fatser calc
EXPORT Tensor* tensor_mmul(Tensor* a, Tensor* b, bool transposed);     // Matrix multiplication when applicable
EXPORT void matmul_backward(Function* f, Tensor* output_grad);
//EXPORT float sum8(__m256 v); // sum a simd register (8 elements)

//other tensor math operations
EXPORT float tensor_dot(Tensor* a, Tensor* b);
EXPORT float tensor_sum(Tensor* t);
EXPORT float tensor_mean(Tensor* t);
EXPORT void tensor_brodcast_inplace(Tensor* target, Tensor* other);
