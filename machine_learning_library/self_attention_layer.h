#pragma once
#include "export.h"

typedef struct  dense_layer;
typedef struct  Tensor;
typedef enum OptimizerType OptimizerType;

EXPORT typedef struct self_attention {
    int seq_len; // amount of tokens in a single input
    int input_dim; //The dimension of each input token vector
    int batch_size; // amount of inout(matrixes) in one pass
    int d_k, d_v;

    dense_layer* W_q; // querys WEIHGTS
    dense_layer* W_k; // KEYS WIEGHTS
    dense_layer* W_v; // VALUES WEIGHTS
    dense_layer* W_o; // THIS LAYER WEIGHTS;
    Tensor* input;
    Tensor* Q; // querys
    Tensor* K; // KEYS
    Tensor* V; // VALUES
    Tensor* scores; // SCORES BEFORE SOFTMAX KEYS * QUERYS /  sqrt(d_k)
    Tensor* A; // SOFTMAX SCORES
    Tensor* attn_output; // SOFTMAX SCORES(A)  * VALUES
    Tensor* output; // FINAL OUTPUT 

    Tensor* dL_dO;
    Tensor* dL_dA;
    Tensor* dL_dV;
    Tensor* dL_dS;
    Tensor* dL_dQ;
    Tensor* dL_dK;
    Tensor* dX_q;
    Tensor* dX_k;
    Tensor* dX_v;
    Tensor* input_grads; // input grads (the output of backprop)

    optimizer* opt;
} self_attention;


EXPORT self_attention* self_attention_create(int seq_len, int input_dim, int d_k, int d_v, int batch_size);
EXPORT void self_attention_forward(self_attention* sa, Tensor* seq);
EXPORT void self_attention_backward(self_attention* sa, Tensor* dL_dY);
EXPORT void set_self_attention_optimizer(self_attention* sa, OptimizerType type);
EXPORT void self_attention_layer_update(self_attention* sa, float learning_rate);
EXPORT void self_attention_free(self_attention* sa);
EXPORT void softmax(Tensor* dest, Tensor* source);
EXPORT void softmax_derivative(Tensor* input, Tensor* softmax_output, Tensor* grads);