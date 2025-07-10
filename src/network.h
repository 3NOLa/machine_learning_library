#pragma once
#include <stdio.h>
#include "general_layer.h"
#include "export.h"


typedef enum OptimizerType OptimizerType;

EXPORT typedef enum {
	MSE,
	MAE,
	Binary_Cross_Entropy,
	Categorical_Cross_Entropy,
	Huber_Loss
}LossType;


EXPORT typedef struct {
	int layerAmount;
	layer** layers;
	int* layersSize;

	float  learnningRate;
	LossType lossFunction;
	float  (*LossFuntionPointer)(struct network*, Tensor*);
	Tensor* (*LossDerivativePointer)(struct network*, Tensor*);
	float  (*train)(struct network*, Tensor*, Tensor*);

	int input_dims;
	int* input_shape;
	LayerType ltype;

	OptimizerType otype;
}network;

EXPORT network* network_create(int layerAmount, int* layersSize, int input_dims, int* input_shape, ActivationType* activations, float  learnningRate, LossType lossFunction, LayerType type);
EXPORT network* network_create_empty();
EXPORT void network_free(network* net);

EXPORT int add_layer(network* net, int layerSize, ActivationType Activationfunc, int input_dim);// add input layer if first layer otherwise put 0
EXPORT int add_created_layer(network* net, layer* l);

EXPORT int set_loss_function(network* net, LossType lossFunction);
EXPORT void set_network_optimizer(network* net, OptimizerType type);

EXPORT void network_train_type(network* net);

EXPORT Tensor* forwardPropagation(network* net, Tensor* data);
EXPORT int backpropagation(network* net, Tensor* predictions, Tensor* targets);
EXPORT void network_update(network* net);
EXPORT void network_zero_grad(network* net);
EXPORT void network_opt_init(network* net, Initializer* init, initializerType type);
EXPORT void network_reset_state(network* net);
EXPORT int save_model(const network* net, const char* cfg_path, const char* weights_path);
EXPORT void load_weights_model(network* net, FILE* wfp);
EXPORT network* load_model(const char* cfg_path, const char* weights_path);

EXPORT float  train(network* net, Tensor* input, Tensor* target);
EXPORT EXPORT void network_training(network* net, Tensor* input, Tensor* target, int epcho,int batch_size);
EXPORT float  rnn_train(network* net, Tensor* input, Tensor* target, int timestamps);

// implemented in loss_Functions.c
EXPORT float  squared_error_net(network* net, Tensor* y_real);
EXPORT Tensor* derivative_squared_error_net(network* net, Tensor* y_real);

EXPORT float  absolute_error_net(network* net, Tensor* y_real);
EXPORT Tensor* derivative_absolute_error_net(network* net, Tensor* y_real);

EXPORT float  absolute_error_net(network* net, Tensor* y_real);
EXPORT Tensor* derivative_absolute_error_net(network* net, Tensor* y_real);

EXPORT float  Categorical_Cross_Entropy_net(network* net, Tensor* y_real);
EXPORT Tensor* derivative_Categorical_Cross_Entropy_net(network* net, Tensor* y_real);

EXPORT float  (*LossTypeMap(LossType function))(network*, Tensor*);
EXPORT Tensor* (*LossTypeDerivativeMap(LossType function))(network*, Tensor*);

EXPORT float loss_active_function(LossType function, Tensor* y_pred, Tensor* y_real);
EXPORT Tensor* loss_derivative_active_function(LossType function, Tensor* y_pred, Tensor* y_real);

