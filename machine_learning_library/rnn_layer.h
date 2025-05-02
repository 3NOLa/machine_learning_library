#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "rnn_neuron.h"

typedef struct {
    int neuronAmount;
    Tensor* output;
    rnn_neuron** neurons;
    ActivationType Activationenum;
	int sequence_length; // =t = timestamp
}rnn_layer;

rnn_layer* rnn_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc);
Tensor* rnn_layer_forward(rnn_layer* rl, Tensor* input);
Tensor* rnn_layer_backward(rnn_layer* rl, Tensor* output_gradients, double learning_rate);
void rnn_layer_reset_state(rnn_layer* rl);
void rnn_layer_free(rnn_layer* rl);