#pragma once
#include <stdlib.h>
#include <stdio.h>
#include "export.h"

#define MAX_TIMESTEPS 128

typedef struct neuron;
typedef struct rnn_neuron;
typedef struct Tensor;
typedef struct optimizer optimizer; 
typedef enum ActivationType;

EXPORT typedef struct lstm_neuron {
    float  short_memory;//cell state
    float  long_memory;
    Tensor* input_history[MAX_TIMESTEPS];
    float  short_memory_history[MAX_TIMESTEPS];
    float  long_memory_history[MAX_TIMESTEPS];//cell state history
    int timestamp;

    rnn_neuron* i_g_r; //input_gate_remember
    rnn_neuron* i_g_p; //input_gate_potinal also known as candidate cell
    rnn_neuron* o_g_r; //output_gate_remember 
    rnn_neuron* f_g; //forget_gate

    optimizer* opt;
} lstm_neuron;

EXPORT lstm_neuron* lstm_neuron_create(int weightslength, ActivationType func);
EXPORT float  lstm_neuron_activation(Tensor* input, lstm_neuron* ln);
EXPORT void lstm_neuron_backward(float  derivative, lstm_neuron* ln, Tensor* input_gradients);
EXPORT void lstm_neuron_update(lstm_neuron* ln, float rl);
EXPORT void lstm_neuron_zero_grad(lstm_neuron* ln);
EXPORT void lstm_neuron_free(lstm_neuron* ln);