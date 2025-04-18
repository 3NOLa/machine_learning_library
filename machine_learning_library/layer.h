#pragma once
#include <stdio.h>
#include "neuron.h"
#include "active_functions.h"

typedef struct {
	int neuronAmount;
	Tensor* output;
	neuron** neurons;
	ActivationType Activationenum;
}layer;

layer* layer_create(int neuronAmount,int neuronDim ,ActivationType Activationfunc);
void layer_removeLastNeuron(layer* l);
void layer_addNeuron(layer* l);
void layer_set_neuronAmount(layer* l,int neuronAmount);
void layer_set_activtion(layer* l, ActivationType Activationfunc);
Tensor* layer_forward(layer* l, Tensor* input);
Tensor* layer_backward(layer* l, Tensor* input_gradients, double learning_rate);
void layer_free(layer* l);
