#pragma once
#include "tensor.h"
#include "active_functions.h"

typedef struct neuron {
    Tensor* weights;
    float  bias;
    Tensor* input;
    float  pre_activation;
    float  output;
    ActivationType Activation;          
    float  (*ActivationFunc)(float  value); 
    float  (*ActivationderivativeFunc)(neuron* );
} neuron;

neuron* neuron_create(int weightslength, ActivationType func);
void neuron_set_ActivationType(neuron* n,ActivationType Activation);
float  neuron_activation(Tensor* input, neuron* n);
Tensor* neuron_backward(float  derivative, neuron* n, float  learning_rate);
void neuron_free(neuron* n);