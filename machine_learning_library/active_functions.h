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
double RELu(double value);
double Sigmoid(double value);
double Tanh(double value);
void (*ActivationTypeMap(ActivationType function))(double value);
