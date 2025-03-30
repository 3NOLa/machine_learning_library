#pragma once
#include "tensor.h"
#include "active_functions.h"

typedef struct neuron neuron;

typedef struct neuron {
    Tensor* weights;
    double bias;
    Tensor* input;
    double pre_activation;
    double output;
    ActivationType Activationenum;          
    double (*ActivationFunc)(double value); 
    double (*ActivationderivativeFunc)(neuron* );
} neuron;

neuron* neuron_create(int weightslength, ActivationType func);
double neuron_activation(Tensor* input, neuron* n);
Tensor* neuron_backward(double derivative, neuron* n, double learning_rate);
void neuron_free(neuron* n);