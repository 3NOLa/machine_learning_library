#pragma once
#include <stdio.h>
#include "layer.h"

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
	double learnningRate;
	LossType lossFunction;
	double (*LossFuntionPointer)(struct network*, Tensor*);
	Tensor* (*LossDerivativePointer)(struct network*, Tensor*);
}network;

network* network_create(int layerAmount, int* layersSize, int input_dim, ActivationType* activations, double learnningRate,LossType lossFunction);
network* network_create_empty();
int add_layer(network* net, int layerSize, ActivationType Activationfunc, int input_dim);// add input layer if first layer otherwise put 0
void network_free(network* net);

Tensor* forwardPropagation(network* net, Tensor* data);
int backpropagation(network* net, Tensor* predictions, Tensor* targets);

double train(network* net, Tensor* input, Tensor* target);

Tensor* derivative_squared_error_net(network* net, Tensor* y_real);
double squared_error_net(network* net, Tensor* y_real);

double (*LossTypeMap(LossType function))(network*, Tensor*);
Tensor* (*LossTypeDerivativeMap(LossType function))(network*, Tensor*);
