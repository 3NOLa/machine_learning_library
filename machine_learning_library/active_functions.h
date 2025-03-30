#pragma once
#define _USE_MATH_DEFINES
#include "tensor.h"
#ifndef ACTIVE_FUNCTIONS_H  
#define ACTIVE_FUNCTIONS_H
#endif
#include <math.h>


typedef struct neuron neuron;

typedef enum {
	RELU,
	LEAKY_RELU,
	SIGMOID,
	TANH,
	LINEAR,
	GELU,
	SWISH
}ActivationType;

double RELu_function(double value);
double RELu_derivative_function(neuron* n);

double leaky_RELu_function(double value);
double leaky_RELu_derivative_function(neuron* n);

double Sigmoid_function(double value);
double Sigmoid_derivative_function(neuron* n);

double Tanh_function(double value);
double Tanh_derivative_function(neuron* n);

double linear_function(double value);
double linear_derivative_function(neuron* n);

double gelu_function(double value);
double gelu_derivative_function(neuron* n);

double swish_function(double value);
double swish_derivative_function(neuron* n);

double (*ActivationTypeMap(ActivationType function))(double); 
double (*ActivationTypeDerivativeMap(ActivationType function))(neuron*);
