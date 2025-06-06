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
EXPORT Tensor* lstm_layer_backward(lstm_layer* ll, Tensor* output_gradients);
EXPORT void lstm_layer_update(lstm_layer* ll, float lr);
EXPORT void lstm_layer_zero_grad(lstm_layer* ll);
EXPORT void lstm_layer_opt_init(lstm_layer* ll, Initializer* init, initializerType type);
EXPORT void lstm_layer_reset_state(lstm_layer* ll);
EXPORT void lstm_layer_free(lstm_layer* ll);