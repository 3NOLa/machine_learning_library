#pragma once
#include <stdlib.h>
#include <stdio.h>

#define MAX_TIMESTEPS 128

typedef struct neuron;
typedef struct rnn_neuron;
typedef struct Tensor;
typedef enum ActivationType;

typedef struct lstm_neuron {
    rnn_neuron* f_g; //forget_gate
    rnn_neuron* i_g_r; //input_gate_remember
    rnn_neuron* i_g_p; //input_gate_potinal also known as candidate cell
    rnn_neuron* o_g_r; //output_gate_remember 

    double short_memory;//cell state
    double long_memory;
    Tensor* input_history[MAX_TIMESTEPS];
    double short_memory_history[MAX_TIMESTEPS];
    double long_memory_history[MAX_TIMESTEPS];//cell state history
    int timestamp;
} lstm_neuron;

lstm_neuron* lstm_neuron_create(int weightslength, ActivationType func, int layer_amount);
double lstm_neuron_activation(Tensor* input, lstm_neuron* ln);
Tensor* lstm_neuron_backward(double derivative, lstm_neuron* ln, double learning_rate);
void lstm_neuron_free(lstm_neuron* ln);