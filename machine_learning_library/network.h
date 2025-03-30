#pragma once
#include "layer.h"
#include <stdio.h>

typedef struct {
	int layerAmount;
	layer** layers;
	int* layersSize;
	double learnningRate;
}network;

network* network_create(int layerAmount, int* layersSize, int input_dim, ActivationType* activations, double learnningRate);
network* network_create_empty();
int add_layer(network* net, int layerSize, ActivationType Activationfunc, int input_dim);// add input layer if first layer otherwise put 0
void network_free(network* net);

Tensor* forwardPropagation(network* net, Tensor* data);
int backpropagation(network* net, Tensor* predictions, Tensor* targets);

double  squared_error(Tensor* y_hat, Tensor* y_real);
Tensor* derivative_squared_error(Tensor* y_hat, Tensor* y_real); // differentiate before summing for every y_hat

double train(network* net, Tensor* input, Tensor* target);

