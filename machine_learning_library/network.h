#pragma once
#include "layer.h"

typedef struct {
	int layerAmount;
	layer** layers;
	int* layersSize;
	double learnningRate;
}network;

network* network_create(int layerAmount, int* layersSize, int input_dim, ActivationType* activations, double learnningRate);
network* network_create_empty();
void add_layer(network* net, int layerSize, ActivationType Activationfunc, int input_dim);// add input layer if first layer otherwise put 0
void network_free(network* net);

Matrix* forwardPropagation(network* net, Matrix* data);
double  squared_error(Matrix* y_hat, Matrix* y_real);
