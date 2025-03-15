#include "neuron.h"

neuron* neuron_create(int weightslength,ActivationType func)
{
	neuron* n = (neuron*)malloc(sizeof(neuron));
	n->Activationenum = func;
	n->ActivationFunc = ActivationTypeMap(func);
	n->weights = matrix_random_create(1, weightslength);
}

double dot_product(Matrix* input,neuron* n)
{
	if ((input->rows != n->weights->rows) || (input->cols != n->weights->cols))
		return 0.0;

	double sum = 0.0;
	for (int i = 0; i < input->cols; i++)
	{
		sum += input->data[i] * n->weights->data[i];
	}
	
	sum += n->bias;
	
	return n->ActivationFunc(sum);
}
