#pragma once
#include <stdio.h>
#include "general_layer.h"

typedef enum {
	MSE,
	MAE,
	Binary_Cross_Entropy,
	Categorical_Cross_Entropy,
	Huber_Loss
}LossType;


typedef struct {
	int layerAmount;
	layer** layers;
	int* layersSize;
	float  learnningRate;
	LossType lossFunction;
	float  (*LossFuntionPointer)(struct network*, Tensor*);
	Tensor* (*LossDerivativePointer)(struct network*, Tensor*);
	float  (*train)(struct network*, Tensor*, Tensor*);
	int input_dims;
	int* input_shape;
	LayerType type;
}network;

network* network_create(int layerAmount, int* layersSize, int input_dims, int* input_shape, ActivationType* activations, float  learnningRate, LossType lossFunction, LayerType type);
network* network_create_empty();
int add_layer(network* net, int layerSize, ActivationType Activationfunc, int input_dim);// add input layer if first layer otherwise put 0
void network_free(network* net);
void network_train_type(network* net);

Tensor* forwardPropagation(network* net, Tensor* data);
int backpropagation(network* net, Tensor* predictions, Tensor* targets);

float  train(network* net, Tensor* input, Tensor* target);
void network_training(network* net, Tensor* input, Tensor* target, int epcho,int batch_size);
float  rnn_train(network* net, Tensor* input, Tensor* target, int timestamps);

// implemented in loss_Functions.c
float  squared_error_net(network* net, Tensor* y_real);
Tensor* derivative_squared_error_net(network* net, Tensor* y_real);

float  absolute_error_net(network* net, Tensor* y_real);
Tensor* derivative_absolute_error_net(network* net, Tensor* y_real);

float  absolute_error_net(network* net, Tensor* y_real);
Tensor* derivative_absolute_error_net(network* net, Tensor* y_real);

float  Categorical_Cross_Entropy_net(network* net, Tensor* y_real);
Tensor* derivative_Categorical_Cross_Entropy_net(network* net, Tensor* y_real);

float  (*LossTypeMap(LossType function))(network*, Tensor*);
Tensor* (*LossTypeDerivativeMap(LossType function))(network*, Tensor*);
