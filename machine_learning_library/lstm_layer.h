#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "lstm_neuron.h"

typedef struct {
    int neuronAmount;
    Tensor* output;
    lstm_neuron** neurons;
    ActivationType Activationenum;
    int sequence_length;
}lstm_layer;

lstm_layer* lstm_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc);
Tensor* lstm_layer_forward(lstm_layer* ll, Tensor* input);
Tensor* lstm_layer_backward(lstm_layer* ll, Tensor* output_gradients, double learning_rate);
void lstm_layer_reset_state(lstm_layer* ll);
void lstm_layer_free(lstm_layer* ll);