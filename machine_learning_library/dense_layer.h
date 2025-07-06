#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "neuron.h"
#include "active_functions.h"
#include "export.h"

typedef struct Initializer Initializer;
typedef struct optimizer optimizer;
typedef enum initializerType initializerType;
typedef enum OptimizerType OptimizerType;

EXPORT typedef struct {
	int neuronAmount;
	
	Tensor* output;
	Tensor* input;
	Tensor* weights;
	Tensor* bias;

	Tensor* input_grad;
	Tensor* grad_weights;
	Tensor* grad_bias;

	ActivationType Activationenum;
	Initializer* init;
	optimizer* opt;
}dense_layer;

EXPORT dense_layer* dense_layer_create(int neuronAmount,int neuronDim ,ActivationType Activationfunc);
EXPORT void layer_set_activtion(dense_layer* l, ActivationType Activationfunc);
EXPORT void dense_layer_forward(dense_layer* l, Tensor* input);
EXPORT void dense_layer_backward(dense_layer* l, Tensor* output_gradients);
EXPORT void dense_layer_update(dense_layer* layer, float learning_rate);
EXPORT void dense_layer_set_optimizer(dense_layer* layer, OptimizerType type);
EXPORT void dense_layer_zero_grad(dense_layer* dl);
EXPORT void dense_layer_opt_init(dense_layer* dl, Initializer* init, initializerType type);
EXPORT void layer_free(dense_layer* l);
EXPORT int save_dense_layer_model(const FILE* wfp, const FILE* cfp, dense_layer* dl);
EXPORT int load_dense_layer_weights_model(const FILE* wfp, dense_layer* dl);

