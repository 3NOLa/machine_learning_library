#pragma once
#include "tensor.h"
#include "active_functions.h"
#include "export.h"

EXPORT typedef struct neuron {
    Tensor* weights;
    float  bias;
    Tensor* input;
    float  pre_activation;
    float  output;
    ActivationType Activation;          
    float  (*ActivationFunc)(float  value); 
    float  (*ActivationderivativeFunc)(neuron* );
} neuron;

EXPORT neuron* neuron_create(int weightslength, ActivationType func);
EXPORT void neuron_set_ActivationType(neuron* n,ActivationType Activation);
EXPORT float  neuron_activation(Tensor* input, neuron* n);
EXPORT Tensor* neuron_backward(float  derivative, neuron* n, float  learning_rate);
EXPORT void neuron_free(neuron* n);