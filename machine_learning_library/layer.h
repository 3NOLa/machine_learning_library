#pragma once
#include "neuron.h"
#include <stdio.h>

typedef struct {
	int neuronAmount;
	neuron** neurons;
	ActivationType Activationenum;
}layer;

layer* layer_create(int neuronAmount,int neuronDim ,ActivationType Activationfunc);
Matrix* layer_forward(layer* l, Matrix* input);
Matrix* layer_backward(layer* l, Matrix* input_gradients, double learning_rate);
void layer_free(layer* l);
