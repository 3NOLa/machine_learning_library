#pragma once
#include <stdio.h>
#include "dense_layer.h"

typedef enum {
	MSE,
	MAE,
	Binary_Cross_Entropy,
	Categorical_Cross_Entropy,
	Huber_Loss
}LossType;


typedef struct {
	int layerAmount;
	dense_layer** layers;
	int* layersSize;
	double learnningRate;
	LossType lossFunction;
	double (*LossFuntionPointer)(struct network*, Tensor*);
	Tensor* (*LossDerivativePointer)(struct network*, Tensor*);
	int input_dims;
	int* input_shape;
}network;

network* network_create(int layerAmount, int* layersSize, int input_dims, int* input_shape, ActivationType* activations, double learnningRate, LossType lossFunction);
network* network_create_empty();
int add_layer(network* net, int layerSize, ActivationType Activationfunc, int input_dim);// add input layer if first layer otherwise put 0
void network_free(network* net);

Tensor* forwardPropagation(network* net, Tensor* data);
int backpropagation(network* net, Tensor* predictions, Tensor* targets);

double train(network* net, Tensor* input, Tensor* target);
void network_training(network* net, Tensor* input, Tensor* target, int epcho,int batch_size);

// implemented in loss_Functions.c
double squared_error_net(network* net, Tensor* y_real);
Tensor* derivative_squared_error_net(network* net, Tensor* y_real);

double absolute_error_net(network* net, Tensor* y_real);
Tensor* derivative_absolute_error_net(network* net, Tensor* y_real);

double absolute_error_net(network* net, Tensor* y_real);
Tensor* derivative_absolute_error_net(network* net, Tensor* y_real);

double Categorical_Cross_Entropy_net(network* net, Tensor* y_real);
Tensor* derivative_Categorical_Cross_Entropy_net(network* net, Tensor* y_real);

double (*LossTypeMap(LossType function))(network*, Tensor*);
Tensor* (*LossTypeDerivativeMap(LossType function))(network*, Tensor*);
