#include "tensor.h"
#include "neuron.h"
#include "rnn_neuron.h"
#include "lstm_neuron.h"
#include "optimizers.h"
#include <math.h>


void optimizer_set(optimizer* op, OptimizerType type)
{
	if (!op) {
		fprintf(stderr, "Error: NULL optimizer in optimizer_set\n");
		return NULL;
	}

	op->type = type;

    switch (type)
    {
    case SGD:
        op->tensor_update = sgd_tensor_update;
        op->float_update = sgd_float_update;
        break;

    case SGDM:
        op->tensor_update = sgdm_tensor_update;
        op->float_update = sgdm_float_update;
        op->args.momentum.velocity = NULL;
        op->args.momentum.fvelocity = 0.0f;
        op->args.momentum.momentum = 0.9f;
        break;

    case NESTEROV:
        op->tensor_update = nesterov_tensor_update;
        op->float_update = nesterov_float_update;
        op->args.nesterov.velocity = NULL;
        op->args.nesterov.fvelocity = 0.0f;
        op->args.nesterov.momentum = 0.9f;
        break;

    case RMSPROP:
        op->tensor_update = rmsprop_tensor_update;
        op->float_update = rmsprop_float_update;
        op->args.rmsprop.avg_sq_grad = NULL;
        op->args.rmsprop.favg_sq_grad = 0.0f;
        op->args.rmsprop.decay = 0.9f;
        op->args.rmsprop.epsilon = 1e-8f;
        break;

    case ADAM:
        op->tensor_update = adam_tensor_update;
        op->float_update = adam_float_update;
        op->args.adam.m = NULL;
        op->args.adam.v = NULL;
        op->args.adam.t = 0;
        op->args.adam.beta1 = 0.9f;
        op->args.adam.beta2 = 0.999f;
        op->args.adam.epsilon = 1e-8f;
        break;

    default:
        fprintf(stderr, "Error: Unknown optimizer type in optimizer_set\n");
        break;
    }
}

void sgd_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args)
{
    if (!data || !grad || data->count != grad->count) {
        fprintf(stderr, "Error: NULL data or grad or size isnt matching in sgd_tensor_update\n");
        return NULL;
    }

    for (int i = 0; i < data->count; i++)
        data->data[i] += lr * grad->data[i];
}

void sgd_float_update(float* data, float* grad, float lr, OptimizerArgs* args)
{
    *data += *grad * lr;
}

void sgdm_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args)
{
    if(!data || !grad || data->count != grad->count){
		fprintf(stderr, "Error: NULL data or grad or size isnt matching in sgdm_tensor_update\n");
		return NULL;
	}

    if (!args->momentum.velocity) {
        args->momentum.velocity = tensor_zero_create(data->dims, data->shape);
        if (!args->momentum.velocity) {
            fprintf(stderr, "Error: coudent mmalloc momentum in sgdm_tensor_update\n");
            return NULL;
        }
    }

    for (int i = 0; i < data->count; i++)
    {
        args->momentum.velocity->data[i] = args->momentum.momentum * args->momentum.velocity->data[i] + (1 - args->momentum.momentum) * grad->data[i];

        data->data[i] += lr * args->momentum.velocity->data[i];
    }

}

void sgdm_float_update(float* data, float* grad, float lr, OptimizerArgs* args)
{
    args->momentum.fvelocity = args->momentum.momentum * args->momentum.fvelocity + (1 - args->momentum.momentum) * (*grad);
    
    *data += lr * args->momentum.fvelocity;
}

void nesterov_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args)
{
    if (!data || !grad || data->count != grad->count) {
        fprintf(stderr, "Error: NULL data or grad or size isnt matching in nesterov_tensor_update\n");
        return NULL;
    }

    if (!args->nesterov.velocity) {
        args->nesterov.velocity = tensor_zero_create(data->dims, data->shape);
        if (!args->nesterov.velocity) {
            fprintf(stderr, "Error: coudent mmalloc momentum in nesterov_tensor_update\n");
            return NULL;
        }
    }

    for (int i = 0; i < data->count; i++)
    {
        float v_prev = args->nesterov.velocity->data[i];

        args->nesterov.velocity->data[i] = args->nesterov.momentum * args->nesterov.velocity->data[i] + lr * grad->data[i];

        data->data[i] += args->nesterov.momentum * v_prev + lr * grad->data[i];
    }
}

void nesterov_float_update(float* data, float* grad, float lr, OptimizerArgs* args)
{
    float v_prev = args->nesterov.fvelocity;

    args->nesterov.fvelocity = args->nesterov.momentum * args->nesterov.fvelocity + lr * (*grad);

    *data += args->nesterov.momentum * v_prev + lr * (*grad);
}

void adam_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args)
{

}

void adam_float_update(float* data, float* grad, float lr, OptimizerArgs* args)
{

}

void rmsprop_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args)
{
    if (!data || !grad || data->count != grad->count) {
        fprintf(stderr, "Error: NULL data or grad or size isnt matching in rmsprop_tensor_update\n");
        return NULL;
    }

    if (!args->rmsprop.avg_sq_grad) {
        args->rmsprop.avg_sq_grad = tensor_zero_create(data->dims, data->shape);
        if (!args->rmsprop.avg_sq_grad) {
            fprintf(stderr, "Error: coudent malloc momentum in rmsprop_tensor_update\n");
            return NULL;
        }
    }

    for (int i = 0; i < data->count; i++)
    {
        args->rmsprop.avg_sq_grad->data[i] = args->rmsprop.decay * args->rmsprop.avg_sq_grad->data[i] + (1 - args->rmsprop.decay) * grad->data[i] * grad->data[i];

        data->data[i] += lr * (grad->data[i] / (sqrtf(args->rmsprop.avg_sq_grad->data[i]) + args->rmsprop.epsilon));
    }
}

void rmsprop_float_update(float* data, float* grad, float lr, OptimizerArgs* args)
{
    args->rmsprop.favg_sq_grad = args->rmsprop.decay * args->rmsprop.favg_sq_grad + (1 - args->rmsprop.decay) * (*grad) * (*grad);

    *data += lr * (*grad / (sqrtf(args->rmsprop.favg_sq_grad) + args->rmsprop.epsilon));
}

void neuron_opt_update(neuron* n, optimizer* opt, float lr)
{
    opt->tensor_update(n->weights, n->grad_weights,lr ,&(opt->args));
    opt->float_update(&(n->bias), &(n->grad_bias), lr, &(opt->args));
}

void rnn_neuron_opt_update(rnn_neuron* rn, optimizer* opt, float lr)
{
    neuron_opt_update(rn->n, opt, lr);
    opt->float_update(&(rn->recurrent_weights), &(rn->grad_recurrent_weights), lr, &(opt->args));
}

void lstm_neuron_opt_update(lstm_neuron* ln, optimizer* opt, float lr)
{
    rnn_neuron_opt_update(ln->f_g, opt, lr);
    rnn_neuron_opt_update(ln->i_g_p, opt, lr);
    rnn_neuron_opt_update(ln->i_g_r, opt, lr);
    rnn_neuron_opt_update(ln->o_g_r, opt, lr);

}