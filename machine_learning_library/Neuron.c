#include "neuron.h"
#include <stdio.h>

neuron* neuron_create(int weightslength, ActivationType func)
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

    n->Activationenum = func;
    n->input = NULL;  // Initialize to NULL, will be set during activation
    n->ActivationFunc = ActivationTypeMap(func);
    n->ActivationderivativeFunc = ActivationTypeDerivativeMap(func);

    n->weights = matrix_random_create(1, weightslength);
    if (!n->weights) {
        fprintf(stderr, "Error: Failed to create weights for neuron\n");
        free(n);
        return NULL;
    }

    // Initialize bias to a random value between -1 and 1
    n->bias = ((double)rand() / RAND_MAX) * 2 - 1;
    n->output = 0.0;  // Initialize output to 0

    return n;
}

double neuron_activation(Matrix* input, neuron* n)
{
    if (!input || !n) {
        fprintf(stderr, "Error: NULL input or neuron in neuron_activation\n");
        return 0.0;
    }

    if (input->cols != n->weights->cols) {
        fprintf(stderr, "Error: Size mismatch in neuron_activation - input cols: %d, weights cols: %d\n",
            input->cols, n->weights->cols);
        return 0.0;
    }

    // Save input for backpropagation
    if (n->input) {
        matrix_free(n->input);
    }

    n->input = matrix_create(input->rows, input->cols);
    if (!n->input) {
        fprintf(stderr, "Error: Failed to create input copy in neuron_activation\n");
        return 0.0;
    }

    if (!matrix_copy(n->input, input)) {
        fprintf(stderr, "Error: Failed to copy input in neuron_activation\n");
        return 0.0;
    }

    // Calculate weighted sum
    double sum = 0.0;
    for (int i = 0; i < input->cols; i++) {
        sum += matrix_get(input,0,i) * matrix_get(n->weights, 0, i);
    }

    // Add bias
    sum += n->bias;

    // Apply activation function and store output
    n->output = n->ActivationFunc(sum);
    
    return n->output;
}

Matrix* neuron_backward(double output_gradient, neuron* n, double learning_rate)
{
    if (!n || !n->input) {
        fprintf(stderr, "Error: NULL neuron or input in neuron_backward\n");
        return NULL;
    }

    // Derivative of activation function w.r.t. its input
    double activation_derivative = n->ActivationderivativeFunc(n->output);

    // Chain rule - gradient flows through activation function
    double pre_activation_gradient = output_gradient * activation_derivative;

    // Create gradients for inputs
    Matrix* input_gradients = matrix_create(1, n->weights->cols);
    if (!input_gradients) {
        fprintf(stderr, "Error: Failed to create input gradients in neuron_backward\n");
        return NULL;
    }

    // Calculate gradients for this neuron's parameters and inputs
    for (int i = 0; i < n->weights->cols; i++) {
        double original_weight = matrix_get(n->weights, 0, i);
        double weight_gradient = pre_activation_gradient * matrix_get(n->input, 0, i);

        // Store input gradient using original weight
        matrix_set(input_gradients, 0, i, pre_activation_gradient * original_weight);

        // Now update weight
        matrix_set(n->weights, 0, i, original_weight + weight_gradient * learning_rate);
    }

    // Gradient for bias: dL/db = dL/dout * dout/dz * dz/db = output_gradient * activation_derivative * 1
    n->bias += pre_activation_gradient * learning_rate;

    return input_gradients;
}

void neuron_free(neuron* n)
{
    if (n) {
        if (n->weights) matrix_free(n->weights);
        if (n->input) matrix_free(n->input);
        free(n);
    }
}