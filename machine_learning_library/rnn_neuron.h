#pragma once
#include <stdlib.h>
#include <stdio.h>
#include "export.h"

#define MAX_TIMESTEPS 128

typedef struct neuron;
typedef struct Tensor;
typedef enum ActivationType;

EXPORT typedef struct rnn_neuron {
    neuron* n;
    float recurrent_weights;
    float grad_recurrent_weights;
    float hidden_state;
    float grad_hidden_state;
    Tensor* input_history[MAX_TIMESTEPS];        
    float  hidden_state_history[MAX_TIMESTEPS];
    int timestamp;
} rnn_neuron;

EXPORT rnn_neuron* rnn_neuron_create(int weightslength, ActivationType func);
EXPORT void rnn_neuron_set_ActivationType(rnn_neuron* rn, ActivationType Activation);
EXPORT float rnn_neuron_activation(Tensor* input, rnn_neuron* rn);
EXPORT void rnn_neuron_backward(float  output_gradient, rnn_neuron* rn, Tensor* input_grads);
EXPORT void rnn_neuron_update(rnn_neuron* rn, float rl);
EXPORT void rnn_neuron_zero_grad(rnn_neuron* rn);
EXPORT void rnn_neuron_free(rnn_neuron* rn);