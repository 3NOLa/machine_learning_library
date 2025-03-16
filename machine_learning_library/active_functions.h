#pragma once
#include "matrix.h"
#include <math.h>

typedef enum {
	RELu,
	Sigmoid,
	Tanh
}ActivationType;

void RELu_layer(Matrix* mat);
void Sigmoid_layer(Matrix* mat);
void Tanh_layer(Matrix* mat);
double RELu_function(double value);
double Sigmoid_function(double value);
double Tanh_function(double value);
void (*ActivationTypeMap(ActivationType function))(double value);
