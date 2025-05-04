#pragma once
#include <stdio.h>
#include "neuron.h"
#include "active_functions.h"

typedef struct {
	int neuronAmount;
	Tensor* output;
	neuron** neurons;
	ActivationType Activationenum;
}dense_layer;

dense_layer* layer_create(int neuronAmount,int neuronDim ,ActivationType Activationfunc);
void layer_removeLastNeuron(dense_layer* l);
void layer_addNeuron(dense_layer* l);
void layer_set_neuronAmount(dense_layer* l,int neuronAmount);
void layer_set_activtion(dense_layer* l, ActivationType Activationfunc);
Tensor* layer_forward(dense_layer* l, Tensor* input);
Tensor* layer_backward(dense_layer* l, Tensor* input_gradients, float  learning_rate);
void layer_free(dense_layer* l);
