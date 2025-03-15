#include "layer.h"

layer* layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc)
{
	layer* L = (layer*)malloc(sizeof(layer));
	L->Activationenum = Activationfunc;
	L->neuronAmount = neuronAmount;
	
	L->neurons = (neuron**)malloc(sizeof(neuron*) * neuronAmount);
	for (int i = 0; i < neuronAmount; i++)
	{
		L->neurons[i] = neuron_create(neuronDim, Activationfunc);
	}

	return L;
}

Matrix* layer_forward(layer* l, Matrix* input)
{
	Matrix* output = matrix_create(1, l->neuronAmount);

	for (int i = 0; i < l->neuronAmount; i++)
	{
		double activation = neuron_activation(input, l->neurons[i]);
		matrix_set(output, 0, i, activation);
	}

	return output;
}

void layer_free(layer* l)
{
	if (l) {
		for (int i = 0; i < l->neuronAmount; i++)
			neuron_free(l->neurons[i]);
		free(l);
	}
}