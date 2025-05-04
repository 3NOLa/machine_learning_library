#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include "export.h"
#ifndef ACTIVE_FUNCTIONS_H  
#define ACTIVE_FUNCTIONS_H
#endif

// Forward declaration of neuron struct
typedef struct neuron neuron;

EXPORT typedef enum {
	RELU,
	LEAKY_RELU,
	SIGMOID,
	TANH,
	LINEAR,
	GELU,
	SWISH
}ActivationType;

EXPORT float  RELu_function(float  value);
EXPORT float  RELu_derivative_function(neuron* n);

EXPORT float  leaky_RELu_function(float  value);
EXPORT float  leaky_RELu_derivative_function(neuron* n);

EXPORT float  Sigmoid_function(float  value);
EXPORT float  Sigmoid_derivative_function(neuron* n);

EXPORT float  Tanh_function(float  value);
EXPORT float  Tanh_derivative_function(neuron* n);

EXPORT float  linear_function(float  value);
EXPORT float  linear_derivative_function(neuron* n);

EXPORT float  gelu_function(float  value);
EXPORT float  gelu_derivative_function(neuron* n);

EXPORT float  swish_function(float  value);
EXPORT float  swish_derivative_function(neuron* n);

EXPORT float  (*ActivationTypeMap(ActivationType function))(float );
EXPORT float  (*ActivationTypeDerivativeMap(ActivationType function))(neuron*);
