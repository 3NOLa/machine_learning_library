#include "general_layer.h"


layer* general_layer_Initialize(LayerType type, int neuronAmount, int neuronDim, ActivationType Activationfunc)
{
	layer* l = (layer*)malloc(sizeof(layer));
	l->type = type;
	
	switch (type)
	{
	case LAYER_DENSE:
		l->params = layer_create(neuronAmount, neuronDim, Activationfunc);
		l->forward = wrapper_dense_forward;
		l->backward = wrapper_dense_backward;
		l->free = layer_free;
		break;
	case LAYER_RNN:
		l->params = rnn_layer_create(neuronAmount, neuronDim, Activationfunc);
		l->forward = wrapper_rnn_forward;
		l->backward = wrapper_rnn_backward;
		l->free = rnn_layer_free;
		break;
	default:
		fprintf(stderr, "Erorr: not a valid in layer_Initialize\n");
		return NULL;
		break;
	}
}

Tensor* wrapper_rnn_forward(layer* base_layer, Tensor* input) {
	rnn_layer* rl = (rnn_layer*)base_layer->params;
	return rnn_layer_forward(rl, input);
}

Tensor* wrapper_rnn_backward(layer* base_layer,Tensor* grad, double learning_rate) {
	rnn_layer* rl = (rnn_layer*)base_layer->params;
	return rnn_layer_backward(rl, grad, learning_rate);
}

Tensor* wrapper_dense_forward(layer* base_layer, Tensor* input) {
	dense_layer* dl = (dense_layer*)base_layer->params;
	return layer_forward(dl, input);
}

Tensor* wrapper_dense_backward(layer* base_layer, Tensor* grad, double learning_rate) {
	dense_layer* dl = (dense_layer*)base_layer->params;
	return layer_backward(dl, grad, learning_rate);
}

void general_layer_free(layer* base_layer)
{
	switch (base_layer->type)
	{
	case LAYER_DENSE:
		base_layer->free((dense_layer*)base_layer->params);
		break;
	case LAYER_RNN:
		base_layer->free((rnn_layer*)base_layer->params);
		break;
	default:
		fprintf(stderr, "Erorr: not a valid in layer_Initialize\n");
		return NULL;
		break;
	}
	free(base_layer);
}

Tensor* get_layer_output(layer* base_layer)
{
	Tensor* output;
	switch (base_layer->type)
	{
	case LAYER_DENSE:
	{
		dense_layer* dl = AS_DENSE(base_layer);
		output = dl->output;
		break;
	}
	case LAYER_RNN:
	{
		rnn_layer* rl = AS_RNN(base_layer);
		output = rl->output;
		break;
	}
	default:
		fprintf(stderr, "Erorr: not a valid in layer_Initialize\n");
		return NULL;
		break;
	}

	return output;
}