#include "general_layer.h"
#include "optimizers.h"

layer* general_layer_Initialize(LayerType type, int neuronAmount, int neuronDim, ActivationType Activationfunc)
{
	layer* l = (layer*)malloc(sizeof(layer));
	l->type = type;
	l->neuronAmount = neuronAmount;
	
	switch (type)
	{
	case LAYER_DENSE:
		l->params = layer_create(neuronAmount, neuronDim, Activationfunc);
		l->forward = wrapper_dense_forward;
		l->backward = wrapper_dense_backward;
		l->update = wrapper_dense_update;
		l->zero_grad = wrapper_dense_zero_grad;
		l->opt_init = wrapper_dense_opt_init;
		l->free = layer_free;
		l->reset_state = NULL;
		break;
	case LAYER_RNN:
		l->params = rnn_layer_create(neuronAmount, neuronDim, Activationfunc);
		l->forward = wrapper_rnn_forward;
		l->backward = wrapper_rnn_backward;
		l->update = wrapper_rnn_update;
		l->zero_grad = wrapper_rnn_zero_grad;
		l->opt_init = wrapper_rnn_opt_init;
		l->free = rnn_layer_free;
		l->reset_state = wrapper_rnn_reset_state;
		break;
	case LAYER_LSTM:
		l->params = lstm_layer_create(neuronAmount, neuronDim, Activationfunc);
		l->forward = wrapper_lstm_forward;
		l->backward = wrapper_lstm_backward;
		l->update = wrapper_lstm_update;
		l->zero_grad = wrapper_lstm_zero_grad;
		l->opt_init = wrapper_lstm_opt_init;
		l->free = lstm_layer_free;
		l->reset_state = wrapper_rnn_reset_state;
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

Tensor* wrapper_rnn_backward(layer* base_layer,Tensor* grad) {
	rnn_layer* rl = (rnn_layer*)base_layer->params;
	return rnn_layer_backward(rl, grad);
}

void wrapper_rnn_update(layer* base_layer, float lr) {
	rnn_layer* rl = (rnn_layer*)base_layer->params;
	rnn_layer_update(rl, lr);
}

void wrapper_rnn_zero_grad(layer* base_layer) {
	rnn_layer* rl = (rnn_layer*)base_layer->params;
	rnn_layer_zero_grad(rl);
}

void wrapper_rnn_reset_state(layer* base_layer) {
	rnn_layer* rl = (rnn_layer*)base_layer->params;
	rnn_layer_reset_state(rl);
}

void wrapper_rnn_opt_init(layer* base_layer, Initializer* init, initializerType type) {
	rnn_layer* rl = (rnn_layer*)base_layer->params;
	rnn_layer_opt_init(rl, init, type);
}

Tensor* wrapper_dense_forward(layer* base_layer, Tensor* input) {
	dense_layer* dl = (dense_layer*)base_layer->params;
	return layer_forward(dl, input);
}

Tensor* wrapper_dense_backward(layer* base_layer, Tensor* grad) {
	dense_layer* dl = (dense_layer*)base_layer->params;
	return layer_backward(dl, grad);
}

void wrapper_dense_update(layer* base_layer, float lr) {
	dense_layer* dl = (dense_layer*)base_layer->params;
	dense_layer_update(dl, lr);
}

void wrapper_dense_zero_grad(layer* base_layer) {
	dense_layer* dl = (dense_layer*)base_layer->params;
	dense_layer_zero_grad(dl);
}

void wrapper_dense_opt_init(layer* base_layer, Initializer* init, initializerType type) {
	dense_layer* dl = (dense_layer*)base_layer->params;
	dense_layer_opt_init(dl, init, type);
}

Tensor* wrapper_lstm_forward(layer* base_layer, Tensor* input)
{
	lstm_layer* ll = (lstm_layer*)base_layer->params;
	return lstm_layer_forward(ll, input);
}

Tensor* wrapper_lstm_backward(layer* base_layer, Tensor* grad)
{
	lstm_layer* ll = (lstm_layer*)base_layer->params;
	return lstm_layer_backward(ll, grad);
}

void wrapper_lstm_update(layer* base_layer, float lr) {
	lstm_layer* ll = (lstm_layer*)base_layer->params;
	lstm_layer_update(ll, lr);
}

void wrapper_lstm_zero_grad(layer* base_layer) {
	lstm_layer* ll = (lstm_layer*)base_layer->params;
	lstm_layer_zero_grad(ll);
}

void wrapper_lstm_reset_state(layer* base_layer)
{
	lstm_layer* ll = (lstm_layer*)base_layer->params;
	lstm_layer_reset_state(ll);
}

void wrapper_lstm_opt_init(layer* base_layer, Initializer* init, initializerType type) {
	lstm_layer* ll = (lstm_layer*)base_layer->params;
	lstm_layer_opt_init(ll, init, type);
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
	case LAYER_LSTM:
		base_layer->free((lstm_layer*)base_layer->params);
		break;
	default:
		fprintf(stderr, "Erorr: not a valid in general_layer_free\n");
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
	case LAYER_LSTM:
	{
		lstm_layer* ll = AS_LSTM(base_layer);
		output = ll->output;
		break;
	}
	default:
		fprintf(stderr, "Erorr: not a valid in get_layer_output\n");
		return NULL;
		break;
	}

	return output;
}


void set_layer_output(layer* base_layer, Tensor* output)
{//maybe a problem here
	
	switch (base_layer->type)
	{
	case LAYER_DENSE:
	{
		dense_layer* dl = AS_DENSE(base_layer);
		dl->output = output;
		break;
	}
	case LAYER_RNN:
	{
		rnn_layer* rl = AS_RNN(base_layer);
		rl->output = output;
		break;
	}
	case LAYER_LSTM:
	{
		lstm_layer* ll = AS_LSTM(base_layer);
		ll->output = output;
		break;
	}
	default:
		fprintf(stderr, "Erorr: not a valid in set_layer_output\n");
		return NULL;
		break;
	}
}

void set_layer_optimizer(layer* base_layer, OptimizerType type)
{
	switch (base_layer->type)
	{
	case LAYER_DENSE:
	{
		dense_layer* dl = AS_DENSE(base_layer);
		for (int i = 0; i < dl->neuronAmount; i++)
			optimizer_set(dl->neurons[i]->opt, type);
		break;
	}
	case LAYER_RNN:
	{
		rnn_layer* rl = AS_RNN(base_layer);
		for (int i = 0; i < rl->neuronAmount; i++)
			optimizer_set(rl->neurons[i]->opt, type);
		break;
	}
	case LAYER_LSTM:
	{
		lstm_layer* ll = AS_LSTM(base_layer);
		for (int i = 0; i < ll->neuronAmount; i++)
			optimizer_set(ll->neurons[i]->opt, type);
		break;
	}
	default:
		fprintf(stderr, "Erorr: not a valid in set_layer_optimizer\n");
		return NULL;
		break;
	}
}

int save_layer_model(const FILE* wfp, const FILE* cfp,const layer* base_layer) {
	
	switch (base_layer->type)
	{
	case LAYER_DENSE:
	{
		dense_layer* dl = AS_DENSE(base_layer);
		save_dense_layer_model(wfp,cfp, dl);
		break;
	}
	case LAYER_RNN:
	{
		rnn_layer* rl = AS_RNN(base_layer);
		save_rnn_layer_model(wfp, cfp, rl);
		break;
	}
	case LAYER_LSTM:
	{
		lstm_layer* ll = AS_LSTM(base_layer);
		save_lstm_layer_model(wfp, cfp, ll);
		break;
	}
	default:
		fprintf(stderr, "Erorr: not a valid type in save_layer_model\n");
		return NULL;
		break;
	}
}

int load_layer_weights_model(const FILE* wfp, const layer* base_layer){
	switch (base_layer->type)
	{
	case LAYER_DENSE:
	{
		dense_layer* dl = AS_DENSE(base_layer);
		load_dense_layer_weights_model(wfp, dl);
		break;
	}
	case LAYER_RNN:
	{
		rnn_layer* rl = AS_RNN(base_layer);
		load_rnn_layer_weights_model(wfp, rl);
		break;
	}
	case LAYER_LSTM:
	{
		lstm_layer* ll = AS_LSTM(base_layer);
		//save_lstm_layer_model(wfp, cfp, ll);
		break;
	}
	default:
		fprintf(stderr, "Erorr: not a valid type in save_layer_model\n");
		return NULL;
		break;
	}
}
