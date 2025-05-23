#pragma once
#include "dense_layer.h"
#include "rnn_layer.h"
#include "lstm_layer.h"
#include "export.h"

#define AS_DENSE(l) ((dense_layer*)((l)->params))
#define AS_RNN(l)   ((rnn_layer*)((l)->params))
#define AS_LSTM(l)   ((lstm_layer*)((l)->params))

EXPORT typedef enum {
    LAYER_DENSE,
    LAYER_RNN,
    LAYER_LSTM,
    // LAYER_CONV
} LayerType;

EXPORT typedef struct Layer {
    LayerType type;
    int neuronAmount;
    void* params;  
    Tensor* (*forward)(struct Layer* layer, Tensor* input);
    Tensor* (*backward)(struct Layer* layer, Tensor* grad);
    void (*update)(struct Layer* layer, float  learning_rate);
    void (*free)(struct Layer* layer);
    void (*zero_grad)(struct Layer* layer);
    //rnn only
    void (*reset_state)(struct layer* base_layer);
}layer;

EXPORT layer* general_layer_Initialize(LayerType type, int neuronAmount, int neuronDim, ActivationType Activationfunc);
EXPORT void general_layer_free(layer* base_layer);
EXPORT Tensor* get_layer_output(layer* base_layer);
EXPORT void set_layer_output(layer* base_layer, Tensor* output);

EXPORT Tensor* wrapper_rnn_forward(layer* base_layer, Tensor* input);
EXPORT Tensor* wrapper_rnn_backward(layer* base_layer, Tensor* grad);
EXPORT void wrapper_rnn_update(layer* base_layer, float lr);
EXPORT void wrapper_rnn_zero_grad(layer* base_layer);
EXPORT void wrapper_rnn_reset_state(layer* base_layer);

EXPORT Tensor* wrapper_dense_forward(layer* base_layer, Tensor* input);
EXPORT Tensor* wrapper_dense_backward(layer* base_layer, Tensor* gra);
EXPORT void wrapper_dense_update(layer* base_layer, float lr);
EXPORT void wrapper_dense_zero_grad(layer* base_layer);

EXPORT Tensor* wrapper_lstm_forward(layer* base_layer, Tensor* input);
EXPORT Tensor* wrapper_lstm_backward(layer* base_layer, Tensor* grad);
EXPORT void wrapper_lstm_update(layer* base_layer, float lr);
EXPORT void wrapper_lstm_zero_grad(layer* base_layer);
EXPORT void wrapper_lstm_reset_state(layer* base_layer);