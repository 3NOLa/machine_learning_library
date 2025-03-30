#pragma once
#include "tensor.h"
#include <math.h>

typedef enum {
	RELU,
	SIGMOID,
	TANH
}ActivationType;

double RELu_function(double value);
double RELu_derivative_function(double value);
double Sigmoid_function(double value);
double Sigmoid_derivative_function(double value);
double Tanh_function(double value);
double Tanh_derivative_function(double value);
void (*ActivationTypeMap(ActivationType function))(double value);
void (*ActivationTypeDerivativeMap(ActivationType function))(double value);
