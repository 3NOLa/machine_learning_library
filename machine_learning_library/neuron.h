#pragma once
#include "matrix.h"
#include "active_functions.h"

typedef struct {
    Matrix* weights;
    double bias;
    Matrix* input;
    double output;
    ActivationType Activationenum;          
    double (*ActivationFunc)(double value); 
    double (*ActivationderivativeFunc)(double value);
} neuron;

neuron* neuron_create(int weightslength, ActivationType func);
double neuron_activation(Matrix* input, neuron* n, int input_row);
Matrix* neuron_backward(double derivative, neuron* n, double learning_rate);
void neuron_free(neuron* n);