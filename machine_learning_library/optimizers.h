#pragma once
#include "export.h"

typedef struct rnn_neuron rnn_neuron;
typedef struct lstm_neuron lstm_neuron;
typedef struct Tensor;
typedef struct neuron;


EXPORT typedef enum {
	SGD, // SCHOLAR GRADINET DECENT
	SGDM,// SGD WITH MOMENTUM
	NESTEROV, // SGD WITH Nesterov Momentum
	ADAM,
	RMSPROP,
}OptimizerType;

EXPORT typedef struct {
	Tensor* velocity;
	float fvelocity;
	float momentum; // beta
} MomentumState;

EXPORT typedef struct {
	Tensor* velocity;
	float fvelocity;
	float momentum;
}NesterovMomentumState;

EXPORT typedef struct {
	Tensor* avg_sq_grad;
	float favg_sq_grad;
	float decay;
	float epsilon;
} RMSPropState;

EXPORT typedef struct {
	Tensor* m;  // 1st moment (mean)
	Tensor* v;  // 2nd moment (variance)
	float fm;
	float fv;
	int t;      // timestep
	float beta1;
	float beta2;
	float epsilon;
} AdamState;

EXPORT typedef union {
	MomentumState momentum;
	NesterovMomentumState nesterov;
	RMSPropState rmsprop;
	AdamState adam;
}OptimizerArgs;

EXPORT typedef struct optimizer{
    OptimizerType type;
	OptimizerArgs args;
	void (*tensor_update)(Tensor*, Tensor*, float, OptimizerArgs*);
	void (*float_update)(float*, float*, float, OptimizerArgs*);
} optimizer;


EXPORT void optimizer_set(optimizer* op, OptimizerType type);

EXPORT void sgd_tensor_update(Tensor* data, Tensor* grad,float lr, OptimizerArgs* args);
EXPORT void sgd_float_update(float* data, float* grad, float lr, OptimizerArgs* args);

EXPORT void sgdm_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args);
EXPORT void sgdm_float_update(float* data, float* grad, float lr, OptimizerArgs* args);

EXPORT void nesterov_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args);
EXPORT void nesterov_float_update(float* data, float* grad, float lr, OptimizerArgs* args);

EXPORT void adam_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args);
EXPORT void adam_float_update(float* data, float* grad, float lr, OptimizerArgs* args);

EXPORT void rmsprop_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args);
EXPORT void rmsprop_float_update(float* data, float* grad, float lr, OptimizerArgs* args);

EXPORT void neuron_opt_update(neuron* n, optimizer* opt, float lr);
EXPORT void rnn_neuron_opt_update(rnn_neuron* rn, optimizer* opt, float lr);
EXPORT void lstm_neuron_opt_update(lstm_neuron* ln, optimizer* opt, float lr);
