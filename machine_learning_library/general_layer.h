#pragma once
#include "dense_layer.h"
#include "rnn_layer.h"

typedef enum {
    LAYER_DENSE,
    LAYER_RNN,
    // LAYER_CONV
} LayerType;

typedef struct Layer {
    LayerType type;
    void* params;  
    Tensor* (*forward)(struct Layer* layer, Tensor* input);
    Tensor* (*backward)(struct Layer* layer, Tensor* grad, double learning_rate);
    void (*free)(struct Layer* layer);
}layer;


Tensor* forward(Layer* base_layer, Tensor* input);
Tensor* backward(Layer* base_layer, Tensor* grad, double learning_rate);
void free(Layer* base_layer);
