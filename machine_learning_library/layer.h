#pragma once
#include "neuron.h"
#include <stdio.h>

typedef struct {
	int neuronAmount;
	neuron** neurons;
	ActivationType Activationenum;
}layer;

layer* layer_create(int neuronAmount,int neuronDim ,ActivationType Activationfunc);
Tensor* layer_forward(layer* l, Tensor* input);
Tensor* layer_backward(layer* l, Tensor* input_gradients, double learning_rate);
void layer_free(layer* l);
