#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "neuron.h"
#include "active_functions.h"
#include "export.h"

typedef struct Initializer Initializer;
typedef enum initializerType initializerType;

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
EXPORT Tensor* layer_backward(dense_layer* l, Tensor* input_gradients);
EXPORT void dense_layer_update(dense_layer* layer, float learning_rate);
EXPORT void dense_layer_zero_grad(dense_layer* dl);
EXPORT void dense_layer_opt_init(dense_layer* dl, Initializer* init, initializerType type);
EXPORT void layer_free(dense_layer* l);
EXPORT int save_dense_layer_model(const FILE* wfp, const FILE* cfp, const dense_layer* dl);
EXPORT int load_dense_layer_weights_model(const FILE* wfp, const dense_layer* dl);

