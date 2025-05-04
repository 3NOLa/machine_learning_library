#pragma once
#include <stdlib.h>
#include <stdio.h>
#include "export.h"

#define MAX_TIMESTEPS 128

typedef struct neuron;
typedef struct rnn_neuron;
typedef struct Tensor;
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
} lstm_neuron;

EXPORT lstm_neuron* lstm_neuron_create(int weightslength, ActivationType func, int layer_amount);
EXPORT float  lstm_neuron_activation(Tensor* input, lstm_neuron* ln);
EXPORT Tensor* lstm_neuron_backward(float  derivative, lstm_neuron* ln, float  learning_rate);
EXPORT void lstm_neuron_free(lstm_neuron* ln);