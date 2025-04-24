#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "rnn_neuron.h"

typedef struct {
    int neuronAmount;
    Tensor* output;
    rnn_neuron** neurons;
    ActivationType Activationenum;
	int sequence_length;
}rnn_layer;

rnn_layer* rnn_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc);
Tensor* rnn_layer_forward(rnn_layer* rl, Tensor* input, int t);
Tensor* rnn_layer_backward(rnn_layer* rl, Tensor* input_gradients, double learning_rate);
void rnn_layer_free(rnn_layer* rl);