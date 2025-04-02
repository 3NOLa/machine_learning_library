#pragma once
#include "tensor.h"

 struct  network;

typedef enum {
	MSE,
	MAE,
	Binary_Cross_Entropy,
	Categorical_Cross_Entropy,
	Huber_Loss
}LossType;

Tensor* derivative_squared_error_net(network* net, Tensor* y_real);
double squared_error_net(network* net, Tensor* y_real);

double (*LossTypeMap(LossType function))(network*, Tensor*);
double (*LossTypeDerivativeMap(LossType function))(network*, Tensor*);