#pragma once
#include "matrix.h"
#include "active_functions.h"

typedef struct {
    Matrix* weights;
    double bias;
    ActivationType Activationenum;          
    double (*ActivationFunc)(double value);        
} neuron;

neuron* neuron_create(int weightslength, ActivationType func);
double dot_product(Matrix* input, neuron* n);