#pragma once
#include "dense_layer.h"
#include "rnn_layer.h"
#include "lstm_layer.h"

#define AS_DENSE(l) ((dense_layer*)((l)->params))
#define AS_RNN(l)   ((rnn_layer*)((l)->params))
#define AS_LSTM(l)   ((lstm_layer*)((l)->params))

typedef enum {
    LAYER_DENSE,
    LAYER_RNN,
    LAYER_LSTM,
    // LAYER_CONV
} LayerType;

typedef struct Layer {
    LayerType type;
    void* params;  
    Tensor* (*forward)(struct Layer* layer, Tensor* input);
    Tensor* (*backward)(struct Layer* layer, Tensor* grad, float  learning_rate);
    void (*free)(struct Layer* layer);
    //rnn only
    void (*reset_state)(struct layer* base_layer);
}layer;

layer* general_layer_Initialize(LayerType type, int neuronAmount, int neuronDim, ActivationType Activationfunc);
void general_layer_free(layer* base_layer);
Tensor* get_layer_output(layer* base_layer);

Tensor* wrapper_rnn_forward(layer* base_layer, Tensor* input);
Tensor* wrapper_rnn_backward(layer* base_layer, Tensor* grad, float  learning_rate);
void wrapper_rnn_reset_state(layer* base_layer);

Tensor* wrapper_dense_forward(layer* base_layer, Tensor* input);
Tensor* wrapper_dense_backward(layer* base_layer, Tensor* grad, float  learning_rate);

Tensor* wrapper_lstm_forward(layer* base_layer, Tensor* input);
Tensor* wrapper_lstm_backward(layer* base_layer, Tensor* grad, float  learning_rate);
void wrapper_lstm_reset_state(layer* base_layer);