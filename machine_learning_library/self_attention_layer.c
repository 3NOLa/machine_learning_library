#include "dense_layer.h"
#include "self_attention_layer.h"
#include "active_functions.h"
#include "optimizers.h"

self_attention* self_attention_create(int seq_len, int input_dim, int d_k, int d_v, int batch_size) {
    self_attention* sa = (self_attention*)malloc(sizeof(self_attention));
    if (!sa) {
        fprintf(stderr, "ERROR: coudnt allocate self attention in self_attention_create");
        return NULL;
    }

    sa->seq_len = seq_len;
    sa->input_dim = input_dim;
    sa->batch_size = batch_size;
    sa->d_k = d_k;
    sa->d_v = d_v;

    sa->W_q = dense_layer_create(d_k, input_dim, LINEAR);
    sa->W_k = dense_layer_create(d_k, input_dim, LINEAR);
    sa->W_v = dense_layer_create(d_k, input_dim, LINEAR);
    sa->W_o = dense_layer_create(input_dim, d_v, LINEAR); 

    // Allocate buffers
    sa->input = tensor_create(3, (int[]) { batch_size, seq_len, d_k });
    sa->Q = tensor_create(3, (int[]) { batch_size, seq_len, d_k });
    sa->K = tensor_create(3, (int[]) { batch_size, seq_len, d_k });
    sa->V = tensor_create(3, (int[]) { batch_size, seq_len, d_v });

    sa->scores = tensor_create(3, (int[]) { batch_size, seq_len, seq_len });
    sa->A = tensor_create(3, (int[]) { batch_size, seq_len, seq_len });

    sa->attn_output = tensor_create(3, (int[]) { batch_size, seq_len, d_v});
    sa->output = tensor_create(3, (int[]) { batch_size, seq_len, input_dim });

    sa->dL_dO = tensor_create(3, (int[]) { batch_size, seq_len, d_v});
    sa->dL_dA = tensor_create(3, (int[]) { batch_size, seq_len, seq_len});
    sa->dL_dV = tensor_create(3, (int[]) { batch_size, seq_len, d_v});
    sa->dL_dS = tensor_create(3, (int[]) { batch_size, seq_len, seq_len });
    sa->dL_dQ = tensor_create(3, (int[]) { batch_size, seq_len, d_k });
    sa->dL_dK = tensor_create(3, (int[]) { batch_size, seq_len, d_k });
    sa->dX_q = tensor_create(3, (int[]) { batch_size, seq_len, input_dim });
    sa->dX_k = tensor_create(3, (int[]) { batch_size, seq_len, input_dim });
    sa->dX_v = tensor_create(3, (int[]) { batch_size, seq_len, input_dim });
    sa->input_grads = tensor_create(3, (int[]) { batch_size, seq_len, input_dim });

    sa->opt = (optimizer*)malloc(sizeof(optimizer));
    optimizer_set(sa->opt, SGD);

    return sa;
}

void self_attention_forward(self_attention* sa, Tensor* input) {
    tensor_copy(sa->input, input);

    dense_layer_forward(sa->W_q, input);
    sa->Q = sa->W_q->output;

    dense_layer_forward(sa->W_k, input);
    sa->K = sa->W_k->output;

    dense_layer_forward(sa->W_v, input);
    sa->V = sa->W_v->output;

    tensor_mmul(sa->Q, sa->K, sa->scores,true);
    tensor_div_scalar_inplace(sa->scores,(float)(1.0 / sqrt(sa->d_k)));
    softmax(sa->A,sa->scores);

    tensor_mmul(sa->A, sa->V, sa->attn_output,false);

    dense_layer_forward(sa->W_o, sa->attn_output);
    sa->output = sa->W_v->output;
}

void self_attention_backward(self_attention* sa, Tensor* dL_dY) {
    dense_layer_backward(sa->W_o, dL_dY); // updates W_o grads
    sa->dL_dO = sa->W_o->input_grad;

    // 2. O = A @ V
    tensor_mmul(sa->dL_dO,sa->V, sa->dL_dA,true); // (B, T, T)
    tensor_mmul(tensor_transpose(sa->A), sa->dL_dO, sa->dL_dV,false); // (B, T, d_k)

    softmax_derivative(sa->dL_dA, sa->A, sa->dL_dS); // (B, T, T)

    float scale = 1.0f / sqrtf((float)sa->d_k);
    tensor_mul_scalar_inplace(sa->dL_dS, scale);

    tensor_mmul(sa->dL_dS, sa->K, sa->dL_dK,false); // (B, T, d_k)
    tensor_mmul(tensor_transpose(sa->dL_dS), sa->Q, sa->dL_dQ,false); // (B, T, d_k)

    dense_layer_backward(sa->W_q, sa->dL_dQ);
    sa->dX_q = sa->W_q->input_grad;

    dense_layer_backward(sa->W_k, sa->dL_dK);
    sa->dX_k = sa->W_k->input_grad;

    dense_layer_backward(sa->W_v, sa->dL_dV);
    sa->dX_v = sa->W_v->input_grad;


    tensor_add_inplace(sa->dX_q, sa->dX_k);
    tensor_add_inplace(sa->dX_q, sa->dX_v);

    sa->input_grads = sa->dX_q; // output gradient
}

void set_self_attention_optimizer(self_attention* sa, OptimizerType type) {
    optimizer_set(sa->opt, type);
}

void self_attention_layer_update(self_attention* sa, float learning_rate) {
    dense_layer_update(sa->W_k, learning_rate);
    dense_layer_update(sa->W_o, learning_rate);
    dense_layer_update(sa->W_q, learning_rate);
    dense_layer_update(sa->W_v, learning_rate);

}



void self_attention_free(self_attention* sa) {
    if(sa->W_q) layer_free(sa->W_q);
    if (sa->W_k) layer_free(sa->W_k);
    if (sa->W_k) layer_free(sa->W_k);
    if (sa->W_o) layer_free(sa->W_o);

    if (sa->input) tensor_free(sa->input);
    if (sa->Q)tensor_free(sa->Q);
    if (sa->K)tensor_free(sa->K);
    if (sa->V)tensor_free(sa->V);

    if (sa->scores)tensor_free(sa->scores);
    if (sa->A)tensor_free(sa->A);

    if (sa->attn_output)tensor_free(sa->attn_output);

    if (sa->output)tensor_free(sa->output);

    if (sa->dL_dO) layer_free(sa->dL_dO);
    if (sa->dL_dK) layer_free(sa->dL_dK);
    if (sa->dL_dO) layer_free(sa->dL_dO);
    if (sa->dL_dQ) layer_free(sa->dL_dQ);

    if (sa->dL_dS) tensor_free(sa->dL_dS);
    if (sa->dL_dV)tensor_free(sa->dL_dV);
    if (sa->dX_k)tensor_free(sa->dX_k);
    if (sa->dX_q)tensor_free(sa->dX_q);
    if (sa->dX_v)tensor_free(sa->dX_v);

    if (sa->input_grads)tensor_free(sa->input_grads);

    free(sa);
}


void softmax(Tensor* dest, Tensor* source) {
    float max = source->data[0];
    int i = 1;
    for (; i < source->count; i++)
        max = fmaxf(max, source->data[i]);

    float sum = 0;
    for (i = 0; i < source->count; i++) {
        float e = expf(source->data[i] - max);
        dest->data[i] = e;
        sum += e;
    }

    for (i = 0; i < source->count; ++i)
        dest->data[i] /= sum;

}

void softmax_derivative(Tensor* input_grads, Tensor* softmax_output, Tensor* grads) {
    float dot = 0.0;
    for (int i = 0; i < softmax_output->count; i++) dot += softmax_output->data[i] * input_grads->data[i];

    for (int i = 0; i < softmax_output->count; i++)
        grads->data[i] = softmax_output->data[i] * (input_grads->data[i] - dot);
}