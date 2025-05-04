#pragma once
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef ACTIVE_FUNCTIONS_H  
#define ACTIVE_FUNCTIONS_H
#endif

// Forward declaration of neuron struct
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

float  RELu_function(float  value);
float  RELu_derivative_function(neuron* n);

float  leaky_RELu_function(float  value);
float  leaky_RELu_derivative_function(neuron* n);

float  Sigmoid_function(float  value);
float  Sigmoid_derivative_function(neuron* n);

float  Tanh_function(float  value);
float  Tanh_derivative_function(neuron* n);

float  linear_function(float  value);
float  linear_derivative_function(neuron* n);

float  gelu_function(float  value);
float  gelu_derivative_function(neuron* n);

float  swish_function(float  value);
float  swish_derivative_function(neuron* n);

float  (*ActivationTypeMap(ActivationType function))(float ); 
float  (*ActivationTypeDerivativeMap(ActivationType function))(neuron*);
