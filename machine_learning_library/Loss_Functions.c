#include "network.h"

double squared_error_net(network* net, Tensor* y_real)
{

    if (!net || !y_real || !net->layers[net->layerAmount - 1]->output) {
        fprintf(stderr, "Error: NULL matrices in squared_error\n");
        return 0.0;
    }

    Tensor* y_hat = net->layers[net->layerAmount - 1]->output;

    double error = 0.0;
    for (int i = 0; i < y_hat->count; i++) {
        double diff = y_real->data[i] - y_hat->data[i];
        error += diff * diff;
    }

    return error / y_hat->count;
}

Tensor* derivative_squared_error_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !net->layers[net->layerAmount - 1]->output) {
        fprintf(stderr, "Error: NULL matrices in derivative_squared_error\n");
        return NULL;
    }

    Tensor* y_hat = net->layers[net->layerAmount - 1]->output;

    Tensor* derivative = tensor_create(y_hat->dims, y_hat->shape);
    if (!derivative) {
        fprintf(stderr, "Error: Failed to create derivative matrix in derivative_squared_error\n");
        return NULL;
    }

    // Derivative is 2*(y_real - y_hat) for MSE
    // Note: This implements -2*(y_hat - y_real) which is equivalent since we're minimizing
    for (int i = 0; i < y_hat->count; i++) {
        derivative->data[i] = 2 * (y_real->data[i] - y_hat->data[i]);
    }

    return derivative;
}

double (*LossTypeMap(LossType function))(network*, Tensor*)
{
    static double (*map[])(network*, Tensor*) = {
        squared_error_net
    };

    return map[function];
}

Tensor* (*LossTypeDerivativeMap(LossType function))(network*, Tensor*)
{
    static double (*map[])(network*, Tensor*) = {
        derivative_squared_error_net
    };

    return map[function];
}