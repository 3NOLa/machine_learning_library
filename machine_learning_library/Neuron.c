#include "neuron.h"
#include "dense_layer.h"
#include "optimizers.h"
#include <stdio.h>
#include <stdlib.h>

neuron* neuron_create(int weightslength, ActivationType Activation)
{
    if (weightslength <= 0) {
        fprintf(stderr, "Error: Invalid weights length %d in neuron_create\n", weightslength);
        return NULL;
    }

    neuron* n = (neuron*)malloc(sizeof(neuron));
    if (!n) {
        fprintf(stderr, "Error: Memory allocation failed for neuron\n");
        return NULL;
    }

    neuron_set_ActivationType(n, Activation);
    n->input = NULL;  // Initialize to NULL, will be set during activation

    // Create a 1D tensor for weights
    n->weights = tensor_random_create(1, (int[]) { weightslength });
    n->grad_weights = tensor_zero_create(1, (int[]) { weightslength });
    n->opt = (optimizer*)malloc(sizeof(optimizer));
    optimizer_set(n->opt, SGD);
    if (!n->weights || !n->grad_weights || !n->opt) {
        fprintf(stderr, "Error: Failed to create weights or grad_weights or instilze optimizer for neuron\n");
        free(n);
        return NULL;
    }


    n->bias = ((float )rand() / RAND_MAX) * 2 - 1;
    n->grad_bias = 0.0;
    n->output = 0.0;  // Initialize output to 0
    n->pre_activation = 0.0;

    return n;
}

void neuron_set_ActivationType(neuron* n, ActivationType Activation)
{
    n->Activation = Activation;
    n->ActivationFunc = ActivationTypeMap(Activation);
    n->ActivationderivativeFunc = ActivationTypeDerivativeMap(Activation);
}

float  neuron_activation(Tensor* input, neuron* n)
{
    if (!input || !n) {
        fprintf(stderr, "Error: NULL input or neuron in neuron_activation\n");
        return 0.0;
    }

    if (input->count != n->weights->count) {
        fprintf(stderr, "Error: Size mismatch in neuron_activation - input count: %d, weights count: %d\n",
            input->count, n->weights->count);
        return 0.0;
    }

    // Save input for backpropagation
    if (n->input) {
        tensor_free(n->input);
    }
    // Create a new tensor with the same dimensions and shape as input
    n->input = tensor_create(input->dims, input->shape);
    if (!n->input) {
        fprintf(stderr, "Error: Failed to create input copy in neuron_activation\n");
        return 0.0;
    }

    if (!tensor_copy(n->input, input)) {
        fprintf(stderr, "Error: Failed to copy input in neuron_activation\n");
        return 0.0;
    }

    // Calculate weighted sum using tensor operations
    float  sum = 0.0;
    for (int i = 0; i < input->count; i++) {
        // Use proper tensor element access functions
        float  input_val = tensor_get_element_by_index(input, i);
        float  weight_val = tensor_get_element_by_index(n->weights, i);
        sum += input_val * weight_val;
    }

    // Add bias
    sum += n->bias;

    n->pre_activation = sum;
    // Apply activation function and store output
    n->output = n->ActivationFunc(sum);

    return n->output;
}

void neuron_backward(float  output_gradient, neuron* n,Tensor* output_gradients)
{
    if (!n || !n->input) {
        fprintf(stderr, "Error: NULL neuron or input in neuron_backward\n");
        return NULL;
    }
    // Derivative of activation function w.r.t. its input
    float  activation_derivative = n->ActivationderivativeFunc(n);

    // Chain rule - gradient flows through activation function
    float  pre_activation_gradient = output_gradient * activation_derivative;

    // Calculate gradients for this neuron's parameters and inputs
    for (int i = 0; i < n->weights->count; i++) {
        int indices[1] = { i };

        // Get original weight using tensor access
        float  original_weight = tensor_get_element(n->weights, indices);

        // Get input value using tensor access
        float  input_val = tensor_get_element(n->input, indices);

        output_gradients->data[i] += pre_activation_gradient * original_weight;

        n->grad_weights->data[i] += pre_activation_gradient * input_val;
    }

    // Gradient for bias: dL/db = dL/dout * dout/dz * dz/db = output_gradient * activation_derivative * 1
    n->grad_bias += pre_activation_gradient;
}

void neuron_update(neuron* n, float lr) {
    neuron_opt_update(n, n->opt, lr);
}

void neuron_zero_grad(neuron* n)
{
    if (!n) return;
    tensor_fill(n->grad_weights, 0.0f);
    n->grad_bias = 0.0f;
}

void neuron_free(neuron* n)
{
    if (n) {
        if (n->weights) tensor_free(n->weights);
        if (n->input) tensor_free(n->input);
        free(n);
    }
}