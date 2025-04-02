#include "network.h"
#include "loss_functions.h"
#include <string.h>

network* network_create(int layerAmount, int* layersSize, int input_dim, ActivationType* activations, double learnningRate)
{
    if (layerAmount <= 0 || !layersSize || !activations || input_dim <= 0) {
        fprintf(stderr, "Error: Invalid parameters in network_create\n");
        return NULL;
    }

    network* net = (network*)malloc(sizeof(network));
    if (!net) {
        fprintf(stderr, "Error: Memory allocation failed for network\n");
        return NULL;
    }

    net->learnningRate = learnningRate;
    net->layerAmount = layerAmount;

    net->layers = (layer**)malloc(sizeof(layer*) * layerAmount);
    if (!net->layers) {
        fprintf(stderr, "Error: Memory allocation failed for layers array\n");
        free(net);
        return NULL;
    }

    net->layersSize = (int*)malloc(sizeof(int) * layerAmount);
    if (!net->layersSize) {
        fprintf(stderr, "Error: Memory allocation failed for layersSize array\n");
        free(net->layers);
        free(net);
        return NULL;
    }

    int lastLayerSize = input_dim;
    for (int i = 0; i < layerAmount; i++) {
        net->layers[i] = layer_create(layersSize[i], lastLayerSize, activations[i]);
        if (!net->layers[i]) {
            fprintf(stderr, "Error: Failed to create layer %d\n", i);

            // Free previously created layers
            for (int j = 0; j < i; j++) {
                layer_free(net->layers[j]);
            }

            free(net->layersSize);
            free(net->layers);
            free(net);
            return NULL;
        }

        net->layersSize[i] = layersSize[i];
        lastLayerSize = layersSize[i];
    }

    return net;
}

network* network_create_empty()
{
    network* net = (network*)malloc(sizeof(network));
    if (!net) {
        fprintf(stderr, "Error: Memory allocation failed for empty network\n");
        return NULL;
    }

    net->learnningRate = 0.0;
    net->layerAmount = 0;
    net->layers = NULL;
    net->layersSize = NULL;

    return net;
}

int add_layer(network* net, int layerSize, ActivationType Activationfunc, int input_dim)
{
    if (!net) {
        fprintf(stderr, "Error: NULL network in add_layer\n");
        return 0;
    }

    if (layerSize <= 0) {
        fprintf(stderr, "Error: Invalid layer size %d in add_layer\n", layerSize);
        return 0;
    }

    // Determine input dimension for the new layer
    int actual_input_dim;
    if (input_dim > 0) {
        actual_input_dim = input_dim;
    }
    else if (net->layerAmount > 0) {
        actual_input_dim = net->layersSize[net->layerAmount - 1];
    }
    else {
        fprintf(stderr, "Error: Cannot determine input dimension for first layer\n");
        return 0;
    }

    // Resize the layersSize array
    int* new_layersSize = (int*)realloc(net->layersSize, sizeof(int) * (net->layerAmount + 1));
    if (!new_layersSize) {
        fprintf(stderr, "Error: Memory reallocation failed for layersSize in add_layer\n");
        return 0;
    }
    net->layersSize = new_layersSize;

    // Resize the layers array
    layer** new_layers = (layer**)realloc(net->layers, sizeof(layer*) * (net->layerAmount + 1));
    if (!new_layers) {
        fprintf(stderr, "Error: Memory reallocation failed for layers in add_layer\n");
        return 0;
    }
    net->layers = new_layers;

    // Create the new layer
    net->layers[net->layerAmount] = layer_create(layerSize, actual_input_dim, Activationfunc);
    if (!net->layers[net->layerAmount]) {
        fprintf(stderr, "Error: Failed to create layer in add_layer\n");
        return 0;
    }

    // Update layer size and count
    net->layersSize[net->layerAmount] = layerSize;
    net->layerAmount++;

    return 1;
}

void network_free(network* net)
{
    if (net) {
        if (net->layers) {
            for (int i = 0; i < net->layerAmount; i++) {
                if (net->layers[i]) {
                    layer_free(net->layers[i]);
                }
            }
            free(net->layers);
        }

        if (net->layersSize) {
            free(net->layersSize);
        }

        free(net);
    }
}

Tensor* forwardPropagation(network* net, Tensor* data)
{
    if (!net || !data) {
        fprintf(stderr, "Error: NULL network or data in forwardPropagation\n");
        return NULL;
    }

    if (net->layerAmount <= 0) {
        fprintf(stderr, "Error: Network has no layers in forwardPropagation\n");
        return NULL;
    }

    Tensor* current_output = NULL;
    Tensor* current_input = data;

    for (int i = 0; i < net->layerAmount; i++) {
        current_output = layer_forward(net->layers[i], current_input);
        if (!current_output) {
            fprintf(stderr, "Error: Layer %d forward propagation failed\n", i);

            // Free previous intermediate outputs
            if (i > 0 && current_input != data) {
                tensor_free(current_input);
            }

            return NULL;
        }

        // Free previous intermediate output (but not the original input)
        if (i > 0 && current_input != data) {
            tensor_free(current_input);
        }

        current_input = current_output;
    }

    return current_output;
}

double squared_error(Tensor* y_hat, Tensor* y_real)
{
    if (!y_hat || !y_real) {
        fprintf(stderr, "Error: NULL matrices in squared_error\n");
        return 0.0;
    }

    //if (y_hat->rows != y_real->rows || y_hat->cols != y_real->cols) {
      //  fprintf(stderr, "Error: Size mismatch in squared_error - prediction: (%d,%d), target: (%d,%d)\n",
        //    y_hat->rows, y_hat->cols, y_real->rows, y_real->cols);
    //    return 0.0;
    //}

    double error = 0.0;
    for (int i = 0; i < y_hat->count; i++) {
        double diff = y_real->data[i] - y_hat->data[i]; //instead i cen do tensor_subtrect
        error += diff * diff; // tensor_mltiply
    }

    return error / y_hat->count; // tensor_mean
}

Tensor* derivative_squared_error(Tensor* y_hat, Tensor* y_real)
{
    if (!y_hat || !y_real) {
        fprintf(stderr, "Error: NULL matrices in derivative_squared_error\n");
        return NULL;
    }

    //if (y_hat->rows != y_real->rows || y_hat->cols != y_real->cols) {
     //   fprintf(stderr, "Error: Size mismatch in derivative_squared_error - prediction: (%d,%d), target: (%d,%d)\n",
      //      y_hat->rows, y_hat->cols, y_real->rows, y_real->cols);
      //  return NULL;
    //}

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

int backpropagation(network* net, Tensor* predictions, Tensor* targets)
{
    if (!net || !predictions || !targets) {
        fprintf(stderr, "Error: NULL parameters in backpropagation\n");
        return 0;
    }

    if (net->layerAmount <= 0) {
        fprintf(stderr, "Error: Network has no layers in backpropagation\n");
        return 0;
    }

    // Calculate error derivatives
    Tensor* output_gradients = derivative_squared_error_net(net, targets);
    if (!output_gradients) {
        fprintf(stderr, "Error: Failed to compute error derivatives in backpropagation\n");
        return 0;
    }

    // Gradients to be passed to each layer
    Tensor* current_gradients = output_gradients;
    Tensor* new_gradients = NULL;

    // Backpropagate through each layer in reverse order
    for (int i = net->layerAmount - 1; i >= 0; i--) {
        new_gradients = layer_backward(net->layers[i], current_gradients, net->learnningRate);
        if (!new_gradients) {
            fprintf(stderr, "Error: Layer %d backpropagation failed\n", i);

            // Free current gradients if they're not the output gradients (which we'll free later)
            if (current_gradients != output_gradients) {
                tensor_free(current_gradients);
            }

            tensor_free(output_gradients);
            return 0;
        }

        // Free previous gradients (except the original output gradients which we'll free after the loop)
        if (current_gradients != output_gradients) {
            tensor_free(current_gradients);
        }

        current_gradients = new_gradients;
    }

    // Free the final gradients and output gradients
    tensor_free(current_gradients);
    tensor_free(output_gradients);

    return 1;
}

double train(network* net, Tensor* input, Tensor* target)
{
    if (!net || !input || !target) {
        fprintf(stderr, "Error: NULL parameters in train\n");
        return -1.0;  // Return negative error to indicate failure
    }

    // Forward pass
    Tensor* predictions = forwardPropagation(net, input);
    if (!predictions) {
        fprintf(stderr, "Error: Forward propagation failed in train\n");
        return -1.0;
    }

    // Calculate error
    double error = squared_error_net(net, target);

    // Backward pass
    if (!backpropagation(net, predictions, target)) {
        fprintf(stderr, "Error: Backpropagation failed in train\n");
        tensor_free(predictions);
        return -1.0;
    }

    // Free resources
    tensor_free(predictions);

    return error;
}