#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "lstm_neuron.h"
#include "export.h"

EXPORT typedef struct {
    int neuronAmount;
    Tensor* output;
    lstm_neuron** neurons;
    ActivationType Activationenum;
    int sequence_length;
}lstm_layer;

EXPORT lstm_layer* lstm_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc);
EXPORT Tensor* lstm_layer_forward(lstm_layer* ll, Tensor* input);
EXPORT Tensor* lstm_layer_backward(lstm_layer* ll, Tensor* output_gradients, float  learning_rate);
EXPORT void lstm_layer_reset_state(lstm_layer* ll);
EXPORT void lstm_layer_free(lstm_layer* ll);