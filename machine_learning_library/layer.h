#pragma once
#include "neuron.h"

typedef struct {
	int neuronAmount;
	neuron** neurons;
	ActivationType Activationenum;
}layer;

layer* layer_create(int neuronAmount,int neuronDim ,ActivationType Activationfunc);
Matrix* layer_forward(layer* l, Matrix* input);  
void layer_free(layer* l);
