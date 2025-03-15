#include "layer.h"

layer* layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc)
{
	layer* L = (layer*)malloc(sizeof(layer));
	L->Activationenum = Activationfunc;
	L->neuronAmount = neuronAmount;
	
	L->neurons = (neuron*)malloc(sizeof(neuron) * neuronAmount);
	for (int i = 0; i < neuronAmount; i++)
	{
		L->neurons[i] = neuron_create(neuronDim, Activationfunc);
	}

	return L;
}