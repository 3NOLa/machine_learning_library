#pragma once
#include "tensor.h"
#include "active_functions.h"
#include "export.h"

typedef struct optimizer optimizer;

EXPORT typedef struct neuron {
    Tensor* weights;
    Tensor* grad_weights;

    float bias;
    float grad_bias;

    Tensor* input;

    float  pre_activation;
    float  output;

    ActivationType Activation;          
    float  (*ActivationFunc)(float  value); 
    float  (*ActivationderivativeFunc)(neuron* );

    optimizer* opt;
} neuron;

EXPORT neuron* neuron_create(int weightslength, ActivationType func);
EXPORT void neuron_set_ActivationType(neuron* n,ActivationType Activation);
EXPORT float neuron_activation(Tensor* input, neuron* n);
EXPORT void neuron_backward(float  derivative, neuron* n, Tensor* output_gradients);
EXPORT void neuron_update(neuron* n, float lr);
EXPORT void neuron_zero_grad(neuron* n);
EXPORT void neuron_free(neuron* n);