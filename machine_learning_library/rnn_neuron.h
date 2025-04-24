#pragma once
#include <stdlib.h>
#include <stdio.h>

#define MAX_TIMESTEPS 128

typedef struct neuron;
typedef struct Tensor;
typedef enum ActivationType;

typedef struct rnn_neuron {
    neuron* n;
    double recurrent_weights;
    double hidden_state;
    Tensor* input_history[MAX_TIMESTEPS];        
    double hidden_state_history[MAX_TIMESTEPS];
} rnn_neuron;

rnn_neuron* rnn_neuron_create(int weightslength, ActivationType func,int layer_amount);
void rnn_neuron_set_ActivationType(rnn_neuron* rn, ActivationType Activation);
double rnn_neuron_activation(Tensor* input, rnn_neuron* rn, int timestamp);
Tensor* rnn_neuron_backward(double derivative, rnn_neuron* rn, double learning_rate,int timestamp);
void rnn_neuron_free(rnn_neuron* rn);