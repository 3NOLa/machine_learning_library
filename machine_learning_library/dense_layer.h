#pragma once
#include <stdio.h>
#include "neuron.h"
#include "active_functions.h"
#include "export.h"

EXPORT typedef struct {
	int neuronAmount;
	Tensor* output;
	neuron** neurons;
	ActivationType Activationenum;
}dense_layer;

EXPORT dense_layer* layer_create(int neuronAmount,int neuronDim ,ActivationType Activationfunc);
EXPORT void layer_removeLastNeuron(dense_layer* l);
EXPORT void layer_addNeuron(dense_layer* l);
EXPORT void layer_set_neuronAmount(dense_layer* l,int neuronAmount);
EXPORT void layer_set_activtion(dense_layer* l, ActivationType Activationfunc);
EXPORT Tensor* layer_forward(dense_layer* l, Tensor* input);
EXPORT Tensor* layer_backward(dense_layer* l, Tensor* input_gradients, float  learning_rate);
EXPORT void layer_free(dense_layer* l);
