#include "classification.h"


float  squared_error_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !get_layer_output(net->layers[net->layerAmount - 1])) {
        fprintf(stderr, "Error: NULL tensors in squared_error\n");
        return 0.0;
    }

    Tensor* y_hat = get_layer_output(net->layers[net->layerAmount - 1]);

    float  error = 0.0;
    for (int i = 0; i < y_hat->count; i++) {
        float  diff = y_real->data[i] - y_hat->data[i];
        error += diff * diff;
    }

    return error / y_hat->count;
}

Tensor* derivative_squared_error_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !get_layer_output(net->layers[net->layerAmount - 1])) {
        fprintf(stderr, "Error: NULL tensors in derivative_squared_error\n");
        return NULL;
    }

    Tensor* y_hat = get_layer_output(net->layers[net->layerAmount - 1]);

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

float  absolute_error_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !get_layer_output(net->layers[net->layerAmount - 1])) {
        fprintf(stderr, "Error: NULL tensors in squared_error\n");
        return 0.0;
    }

    Tensor* y_hat = get_layer_output(net->layers[net->layerAmount - 1]);

    float  error = 0.0;
    for (int i = 0; i < y_hat->count; i++) {
        error += (y_real->data[i] - y_hat->data[i] > 0)? y_real->data[i] - y_hat->data[i] : (y_real->data[i] - y_hat->data[i]) * (-1);
    }

    return error / y_hat->count;
}

Tensor* derivative_absolute_error_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !get_layer_output(net->layers[net->layerAmount - 1])) {
        fprintf(stderr, "Error: NULL tensors in derivative_squared_error\n");
        return NULL;
    }

    Tensor* y_hat = get_layer_output(net->layers[net->layerAmount - 1]);
    float  epsilon = 1e-15;  // Small constant to prevent division by zero
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

float  Categorical_Cross_Entropy_net(network* net, Tensor* y_real)
{
    int real_class = get_predicted_class(y_real);

    Tensor* y_hat = get_layer_output(net->layers[net->layerAmount - 1]);
    int pred_class = get_predicted_class(y_hat);

    float  loss = -log(y_hat->data[real_class]);

    return loss;
}

Tensor* derivative_Categorical_Cross_Entropy_net(network* net, Tensor* y_real)
{
    if (!net || !y_real || !get_layer_output(net->layers[net->layerAmount - 1])) {
        fprintf(stderr, "Error: NULL tensors in derivative_Categorical_Cross_Entropy_net\n");
        return NULL;
    }

    float  epsilon = 1e-15;  // Small constant to prevent division by zero
    Tensor* y_hat = get_layer_output(net->layers[net->layerAmount - 1]);
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
            float  y_hat_val = fmax(y_hat->data[i], epsilon); // prevent log(0)
            derivative->data[i] = -1.0 / y_hat_val;
        }
        else
            derivative->data[i] = epsilon;
    }

    return derivative;
}

float  (*LossTypeMap(LossType function))(network*, Tensor*)
{
    static float  (*map[])(network*, Tensor*) = {
        squared_error_net,
        absolute_error_net,
        NULL,
        Categorical_Cross_Entropy_net
    };

    return map[function];
}

Tensor* (*LossTypeDerivativeMap(LossType function))(network*, Tensor*)
{
    static float  (*map[])(network*, Tensor*) = {
        derivative_squared_error_net,
        derivative_absolute_error_net,
        NULL,
        derivative_Categorical_Cross_Entropy_net
    };

    return map[function];
}

float loss_active_function(LossType function, Tensor* y_pred, Tensor* y_real)
{
    float  (*LossFuntionPointer)(struct network*, Tensor*) = LossTypeMap(function);

    network net;
    layer l;
    l.type = LAYER_DENSE;
    dense_layer d;
    dense_layer* dl = &d;
    dl->output = y_pred;
    l.params = dl;
    layer* layers[1] = { &l };

    net.layers = layers;
    net.layerAmount = 1;

    return LossFuntionPointer(&net, y_real);
}

Tensor* loss_derivative_active_function(LossType function, Tensor* y_pred, Tensor* y_real)
{
    Tensor* (*LossDerivativePointer)(struct network*, Tensor*) = LossTypeDerivativeMap(function);

    network net;
    layer l;
    l.type = LAYER_DENSE;
    dense_layer d;
    dense_layer* dl = &d;
    dl->output = y_pred;
    l.params = dl;
    layer* layers[1] = { &l };

    net.layers = layers;
    net.layerAmount = 1;

    return LossDerivativePointer(&net, y_real);
}

