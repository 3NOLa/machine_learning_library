#include "neuron.h"
#include <stdio.h>

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
    n->weights = tensor_random_create(2, (int[]) { 1, weightslength });
    if (!n->weights) {
        fprintf(stderr, "Error: Failed to create weights for neuron\n");
        free(n);
        return NULL;
    }

    // Initialize bias to a random value between -1 and 1
    n->bias = ((double)rand() / RAND_MAX) * 2 - 1;
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

double neuron_activation(Tensor* input, neuron* n)
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
    double sum = 0.0;
    for (int i = 0; i < input->count; i++) {
        // Use proper tensor element access functions
        double input_val = tensor_get_element_by_index(input, i);
        double weight_val = tensor_get_element_by_index(n->weights, i);
        sum += input_val * weight_val;
    }

    // Add bias
    sum += n->bias;

    n->pre_activation = sum;
    // Apply activation function and store output
    n->output = n->ActivationFunc(sum);

    return n->output;
}

Tensor* neuron_backward(double output_gradient, neuron* n, double learning_rate)
{
    if (!n || !n->input) {
        fprintf(stderr, "Error: NULL neuron or input in neuron_backward\n");
        return NULL;
    }
    // Derivative of activation function w.r.t. its input
    double activation_derivative = n->ActivationderivativeFunc(n);

    // Chain rule - gradient flows through activation function
    double pre_activation_gradient = output_gradient * activation_derivative;

    // Create gradients for inputs with the same shape as weights
    Tensor* input_gradients = tensor_create(2, n->weights->shape);
    if (!input_gradients) {
        fprintf(stderr, "Error: Failed to create input gradients in neuron_backward\n");
        return NULL;
    }

    // Calculate gradients for this neuron's parameters and inputs
    for (int i = 0; i < n->weights->count; i++) {
        int indices[2] = {0, i };

        // Get original weight using tensor access
        double original_weight = tensor_get_element(n->weights, indices);

        // Get input value using tensor access
        double input_val = tensor_get_element(n->input, indices);

        // Calculate weight gradient
        double weight_gradient = pre_activation_gradient * input_val;

        // Store input gradient using original weight
        tensor_set(input_gradients, indices, pre_activation_gradient * original_weight);

        // Now update weight
        tensor_set(n->weights, indices, original_weight + weight_gradient * learning_rate);
    }

    // Gradient for bias: dL/db = dL/dout * dout/dz * dz/db = output_gradient * activation_derivative * 1
    n->bias += pre_activation_gradient * learning_rate;

    return input_gradients;
}

void neuron_free(neuron* n)
{
    if (n) {
        if (n->weights) tensor_free(n->weights);
        if (n->input) tensor_free(n->input);
        free(n);
    }
}