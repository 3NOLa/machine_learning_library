#pragma once
#include "export.h"

EXPORT typedef enum {
	RELU,
	LEAKY_RELU,
	SIGMOID,
	TANH,
	LINEAR,
	GELU,
	SWISH
}ActivationType;


EXPORT Tensor* RELu_function(Tensor* t);
EXPORT void RELu_derivative_function(Function* f, Tensor* output_grad);

EXPORT Tensor* leaky_RELu_function(Tensor* t);
EXPORT void leaky_RELu_derivative_function(Function* f, Tensor* output_grad);

EXPORT Tensor* Sigmoid_function(Tensor* t);
EXPORT void Sigmoid_derivative_function(Function* f, Tensor* output_grad);

EXPORT Tensor* Tanh_function(Tensor* t);
EXPORT void Tanh_derivative_function(Function* f, Tensor* output_grad);

EXPORT Tensor* linear_function(Tensor* t);
EXPORT void linear_derivative_function(Function* f, Tensor* output_grad);

EXPORT Tensor* gelu_function(Tensor* t);
EXPORT void gelu_derivative_function(Function* f, Tensor* output_grad);

EXPORT Tensor* swish_function(Tensor* t);
EXPORT void swish_derivative_function(Function* f, Tensor* output_grad);
