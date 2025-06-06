#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "rnn_neuron.h"
#include "export.h"

EXPORT typedef struct {
    int neuronAmount;
    Tensor* output;
    rnn_neuron** neurons;
    ActivationType Activationenum;
	int sequence_length; // =t = timestamp
}rnn_layer;

EXPORT rnn_layer* rnn_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc);
EXPORT Tensor* rnn_layer_forward(rnn_layer* rl, Tensor* input);
EXPORT Tensor* rnn_layer_backward(rnn_layer* rl, Tensor* output_gradients);
EXPORT void rnn_layer_update(rnn_layer* rl, float lr);
EXPORT void rnn_layer_zero_grad(rnn_layer* rl);
EXPORT void rnn_layer_opt_init(rnn_layer* rl, Initializer* init, initializerType type);
EXPORT void rnn_layer_reset_state(rnn_layer* rl);
EXPORT void rnn_layer_free(rnn_layer* rl);