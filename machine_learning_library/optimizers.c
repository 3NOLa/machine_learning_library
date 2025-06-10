#include "tensor.h"
#include "neuron.h"
#include "rnn_neuron.h"
#include "lstm_neuron.h"
#include "optimizers.h"
#include <immintrin.h> 
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
        op->args.adam.fm = 0;
        op->args.adam.fv = 0;
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
        return;
    }

    tensor_multiply_scalar_exsting(data, grad, lr);
}

void sgd_float_update(float* data, float* grad, float lr, OptimizerArgs* args)
{
    *data += *grad * lr;
}

void sgdm_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args)
{
    if(!data || !grad || data->count != grad->count){
		fprintf(stderr, "Error: NULL data or grad or size isnt matching in sgdm_tensor_update\n");
		return;
	}

    if (!args->momentum.velocity) {
        args->momentum.velocity = tensor_zero_create(data->dims, data->shape);
        if (!args->momentum.velocity) {
            fprintf(stderr, "Error: coudent mmalloc momentum in sgdm_tensor_update\n");
            return;
        }
    }
    __m256 vm = _mm256_set1_ps(args->momentum.momentum);
    __m256 vmn = _mm256_set1_ps((1 - args->momentum.momentum));
    __m256 vlr = _mm256_set1_ps(lr);
    int i = 0;
    for (; i < data->count - 8; i+=8)
    {
        __m256 amv = _mm256_loadu_ps(&args->momentum.velocity->data[i]);
        __m256 vg = _mm256_loadu_ps(&grad->data[i]);
        __m256 vr = _mm256_add_ps(_mm256_mul_ps(vm, amv), _mm256_mul_ps(vmn, vg));

        __m256 vd = _mm256_loadu_ps(&data->data[i]);
        __m256 vf = _mm256_fmadd_ps(vlr, vr, vd);// (a * b) + c

        _mm256_storeu_ps(&args->momentum.velocity->data[i], vr);
        _mm256_storeu_ps(&data->data[i], vf);
    }

    for (; i < data->count; i++)
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
        return;
    }

    if (!args->nesterov.velocity) {
        args->nesterov.velocity = tensor_zero_create(data->dims, data->shape);
        if (!args->nesterov.velocity) {
            fprintf(stderr, "Error: coudent mmalloc momentum in nesterov_tensor_update\n");
            return;
        }
    }

    __m256 vm = _mm256_set1_ps(args->nesterov.momentum);
    __m256 vlr = _mm256_set1_ps(lr);
    int i = 0;
    for (; i < data->count - 8; i += 8)
    {
        __m256 anv = _mm256_loadu_ps(&args->nesterov.velocity->data[i]);
        __m256 vg = _mm256_loadu_ps(&grad->data[i]);
        __m256 vr = _mm256_add_ps(_mm256_mul_ps(vm, anv), _mm256_mul_ps(vg, vlr));


        __m256 vd = _mm256_loadu_ps(&data->data[i]);
        __m256 vf = _mm256_add_ps(_mm256_fmadd_ps(vm, anv, vr), vd);

        _mm256_storeu_ps(&args->nesterov.velocity->data[i], vr);
        _mm256_storeu_ps(&data->data[i], vf);
    }

    for (; i < data->count; i++)
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
    if (!data || !grad || data->count != grad->count) {
        fprintf(stderr, "Error: NULL data or grad or size isnt matching in adam_tensor_update\n");
        return;
    }

    if (!args->adam.m || !args->adam.v) {
        args->adam.m = tensor_zero_create(data->dims, data->shape);
        args->adam.v = tensor_zero_create(data->dims, data->shape);
        if (!args->adam.m || !args->adam.v) {
            fprintf(stderr, "Error: coudent mmalloc momentum in adam_tensor_update\n");
            return;
        }
    }

    args->adam.t++;

    __m256 vb1 = _mm256_set1_ps(args->adam.beta1);
    __m256 vb2 = _mm256_set1_ps(args->adam.beta2);
    __m256 ve = _mm256_set1_ps(args->adam.epsilon);
    __m256 vb1n = _mm256_set1_ps((1.0f - args->adam.beta1));
    __m256 vb2n = _mm256_set1_ps((1.0f - args->adam.beta2));
    __m256 vpf1 = _mm256_set1_ps((1.0f - powf(args->adam.beta1, args->adam.t)));
    __m256 vpf2 = _mm256_set1_ps((1.0f - powf(args->adam.beta2, args->adam.t)));
    __m256 vlr = _mm256_set1_ps(lr);
    int i = 0;
    for (; i < data->count - 8; i += 8)
    {
        __m256 vmd = _mm256_loadu_ps(&args->adam.m->data[i]);
        __m256 vvd = _mm256_loadu_ps(&args->adam.v->data[i]);
        __m256 vg = _mm256_loadu_ps(&grad->data[i]);

        __m256 vf1 = _mm256_fmadd_ps(vb1, vmd, _mm256_mul_ps(vb1n, vg));

        __m256 vg2 = _mm256_mul_ps(vg, vg);
        __m256 vf2 = _mm256_fmadd_ps(vb2, vvd, _mm256_mul_ps(vb2n, vg2));

        __m256 vmh = _mm256_div_ps(vf1, vpf1);
        __m256 vvh = _mm256_div_ps(vf2, vpf2);

        __m256 vd = _mm256_loadu_ps(&data->data[i]);
        __m256 vfinal = _mm256_add_ps(_mm256_div_ps(_mm256_mul_ps(vlr,vmh), _mm256_add_ps(_mm256_sqrt_ps(vvh),ve)), vd);

        _mm256_storeu_ps(&args->adam.m->data[i], vf1);
        _mm256_storeu_ps(&args->adam.v->data[i], vf2);
        _mm256_storeu_ps(&data->data[i], vfinal);
    }

    for (; i < data->count; i++)
    {

        args->adam.m->data[i] = args->adam.beta1 * args->adam.m->data[i] + (1.0f - args->adam.beta1) * grad->data[i];
        args->adam.v->data[i] = args->adam.beta2 * args->adam.v->data[i] + (1.0f - args->adam.beta2) * grad->data[i] * grad->data[i];

        float m_hat = args->adam.m->data[i] / (1.0f - powf(args->adam.beta1, args->adam.t));
        float v_hat = args->adam.v->data[i] / (1.0f - powf(args->adam.beta2, args->adam.t));

        data->data[i] += (lr * m_hat) / (sqrtf(v_hat) + args->adam.epsilon);
    }
}

void adam_float_update(float* data, float* grad, float lr, OptimizerArgs* args)
{
    args->adam.t++;

    args->adam.fm = args->adam.beta1 * args->adam.fm + (1.0f - args->adam.beta1) * (*grad);
    args->adam.fv = args->adam.beta2 * args->adam.fv + (1.0f - args->adam.beta2) * (*grad) * (*grad);

    float m_hat = args->adam.fm / (1.0f - powf(args->adam.beta1, args->adam.t));
    float v_hat = args->adam.fv / (1.0f - powf(args->adam.beta2, args->adam.t));

    *data += (lr * m_hat) / (sqrtf(v_hat) + args->adam.epsilon);
}

void rmsprop_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args)
{
    if (!data || !grad || data->count != grad->count) {
        fprintf(stderr, "Error: NULL data or grad or size isnt matching in rmsprop_tensor_update\n");
        return;
    }

    if (!args->rmsprop.avg_sq_grad) {
        args->rmsprop.avg_sq_grad = tensor_zero_create(data->dims, data->shape);
        if (!args->rmsprop.avg_sq_grad) {
            fprintf(stderr, "Error: coudent malloc momentum in rmsprop_tensor_update\n");
            return;
        }
    }


    __m256 vard = _mm256_set1_ps(args->rmsprop.decay); 
    __m256 vare = _mm256_set1_ps(args->rmsprop.epsilon);
    __m256 vardn = _mm256_set1_ps((1 - args->rmsprop.decay)); 
    __m256 vlr = _mm256_set1_ps(lr);
    int i = 0;
    for (; i < data->count - 8; i += 8)
    {
        __m256 varad = _mm256_loadu_ps(&args->rmsprop.avg_sq_grad->data[i]);
        __m256 vg = _mm256_load_ps(&grad->data[i]);
        __m256 vg2 = _mm256_mul_ps(vg, vg);
        
        __m256 vr = _mm256_mul_ps(vard, varad);
        __m256 vf = _mm256_fmadd_ps(vardn, vg2, vr);

        __m256 vd = _mm256_load_ps(&data->data[i]);
        __m256 vupdate_term = _mm256_mul_ps(vlr, _mm256_div_ps(vg, _mm256_sqrt_ps(_mm256_add_ps(vf, vare))));
        __m256 vfinal = _mm256_add_ps(vd, vupdate_term);

        _mm256_storeu_ps(&args->rmsprop.avg_sq_grad->data[i], vf);
        _mm256_storeu_ps(&data->data[i], vfinal);
    }

    for (; i < data->count; i++)
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