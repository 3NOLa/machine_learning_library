#include "classification.h"


double squared_error_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !net->layers[net->layerAmount - 1]->output) {
        fprintf(stderr, "Error: NULL tensors in squared_error\n");
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
        fprintf(stderr, "Error: NULL tensors in derivative_squared_error\n");
        return NULL;
    }

    Tensor* y_hat = net->layers[net->layerAmount - 1]->output;

    Tensor* derivative = tensor_create(y_hat->dims, y_hat->shape);
    if (!derivative) {
        fprintf(stderr, "Error: Failed to create derivative tensor in derivative_squared_error\n");
        return NULL;
    }

    // Derivative is 2*(y_real - y_hat) for MSE
    // Note: This implements -2*(y_hat - y_real) which is equivalent since we're minimizing
    for (int i = 0; i < y_hat->count; i++) {
        derivative->data[i] = 2 * (y_real->data[i] - y_hat->data[i]);
    }

    return derivative;
}

double absolute_error_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !net->layers[net->layerAmount - 1]->output) {
        fprintf(stderr, "Error: NULL tensors in squared_error\n");
        return 0.0;
    }

    Tensor* y_hat = net->layers[net->layerAmount - 1]->output;

    double error = 0.0;
    for (int i = 0; i < y_hat->count; i++) {
        error += (y_real->data[i] - y_hat->data[i] > 0)? y_real->data[i] - y_hat->data[i] : (y_real->data[i] - y_hat->data[i]) * (-1);
    }

    return error / y_hat->count;
}

Tensor* derivative_absolute_error_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !net->layers[net->layerAmount - 1]->output) {
        fprintf(stderr, "Error: NULL tensors in derivative_squared_error\n");
        return NULL;
    }

    Tensor* y_hat = net->layers[net->layerAmount - 1]->output;
    double epsilon = 1e-15;  // Small constant to prevent division by zero
    Tensor* derivative = tensor_create(y_hat->dims, y_hat->shape);
    if (!derivative) {
        fprintf(stderr, "Error: Failed to create derivative tensor in derivative_squared_error\n");
        return NULL;
    }

    for (int i = 0; i < y_hat->count; i++) {
        if (y_real->data[i] > y_hat->data[i])
            derivative->data[i] = 1;
        else if (y_real->data[i] < y_hat->data[i])
            derivative->data[i] = -1;
        else
            derivative->data[i] = epsilon;
    }
    return derivative;
}

double Categorical_Cross_Entropy_net(network* net, Tensor* y_real)
{
    int real_class = get_predicted_class(y_real);

    Tensor* y_hat = net->layers[net->layerAmount - 1]->output;
    int pred_class = get_predicted_class(y_hat);

    double loss = -log(y_hat->data[real_class]);

    return loss;
}

Tensor* derivative_Categorical_Cross_Entropy_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !net->layers[net->layerAmount - 1]->output) {
        fprintf(stderr, "Error: NULL tensors in derivative_Categorical_Cross_Entropy_net\n");
        return NULL;
    }

    double epsilon = 1e-15;  // Small constant to prevent division by zero
    Tensor* y_hat = net->layers[net->layerAmount - 1]->output;
    Tensor* derivative = tensor_create(y_hat->dims, y_hat->shape);
    if (!derivative) {
        fprintf(stderr, "Error: Failed to create derivative tensor in derivative_Categorical_Cross_Entropy_net\n");
        return NULL;
    }

    int real_class = get_predicted_class(y_real);
    for (int i = 0; i < y_hat->count; i++)
    {
        if (i == real_class)
        {
            double y_hat_val = fmax(y_hat->data[i], epsilon); // prevent log(0)
            derivative->data[i] = -1.0 / y_hat_val;
        }
        else
            derivative->data[i] = epsilon;
    }

    return derivative;
}

double (*LossTypeMap(LossType function))(network*, Tensor*)
{
    static double (*map[])(network*, Tensor*) = {
        squared_error_net,
        absolute_error_net,
        NULL,
        Categorical_Cross_Entropy_net
    };

    return map[function];
}

Tensor* (*LossTypeDerivativeMap(LossType function))(network*, Tensor*)
{
    static double (*map[])(network*, Tensor*) = {
        derivative_squared_error_net,
        derivative_absolute_error_net,
        NULL,
        derivative_Categorical_Cross_Entropy_net
    };

    return map[function];
}