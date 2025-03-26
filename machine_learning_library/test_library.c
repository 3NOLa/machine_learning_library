#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "neuron.h"
#include "layer.h"
#include "network.h"
#include "active_functions.h"


void print_network_weights(network* net) {
    printf("Network weights:\n");
    for (int l = 0; l < net->layerAmount; l++) {
        printf("Layer %d:\n", l);
        for (int n = 0; n < net->layersSize[l]; n++) {
            neuron* neuron = net->layers[l]->neurons[n];
            printf("  Neuron %d: bias=%.4f weights=[", n, neuron->bias);
            for (int w = 0; w < neuron->weights->cols; w++) {
                printf("%.4f", neuron->weights->data[w]);
                if (w < neuron->weights->cols - 1) printf(", ");
            }
            printf("]\n");
        }
    }
}

// Function to create a simple XOR dataset
void create_xor_dataset(Matrix** inputs, Matrix** outputs, int num_samples) {
    // Create inputs: [0,0], [0,1], [1,0], [1,1]
    *inputs = matrix_create(num_samples, 2);
    *outputs = matrix_create(num_samples, 1);

    if (!*inputs || !*outputs) {
        fprintf(stderr, "Failed to create dataset matrices\n");
        return;
    }

    // XOR truth table
    matrix_set(*inputs, 0, 0, 0.0); matrix_set(*inputs, 0, 1, 0.0); matrix_set(*outputs, 0, 0, 0.0);
    matrix_set(*inputs, 1, 0, 0.0); matrix_set(*inputs, 1, 1, 1.0); matrix_set(*outputs, 1, 0, 1.0);
    matrix_set(*inputs, 2, 0, 1.0); matrix_set(*inputs, 2, 1, 0.0); matrix_set(*outputs, 2, 0, 1.0);
    matrix_set(*inputs, 3, 0, 1.0); matrix_set(*inputs, 3, 1, 1.0); matrix_set(*outputs, 3, 0, 0.0);
}

// Function to test matrix operations
void test_matrix_operations2() {
    printf("\n===== Testing Matrix Operations =====\n");

    // Test matrix creation
    Matrix* a = matrix_random_create(2, 3);
    Matrix* b = matrix_random_create(3, 2);
    Matrix* c = matrix_identity_create(3);

    if (!a || !b || !c) {
        fprintf(stderr, "Matrix creation failed\n");
        return;
    }

    printf("Matrix A (2x3, random):\n");
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            printf("%.4f ", matrix_get(a, i, j));
        }
        printf("\n");
    }

    printf("\nMatrix B (3x2, random):\n");
    for (int i = 0; i < b->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            printf("%.4f ", matrix_get(b, i, j));
        }
        printf("\n");
    }

    printf("\nMatrix C (3x3, identity):\n");
    for (int i = 0; i < c->rows; i++) {
        for (int j = 0; j < c->cols; j++) {
            printf("%.4f ", matrix_get(c, i, j));
        }
        printf("\n");
    }

    // Test matrix multiplication
    Matrix* ab = matrix_mul(a, b);
    if (ab) {
        printf("\nMatrix A * B (2x2):\n");
        for (int i = 0; i < ab->rows; i++) {
            for (int j = 0; j < ab->cols; j++) {
                printf("%.4f ", matrix_get(ab, i, j));
            }
            printf("\n");
        }
        matrix_free(ab);
    }
    else {
        printf("Matrix multiplication failed\n");
    }

    // Test matrix scalar operations
    Matrix* a_scaled = matrix_scalar_mul(a, 2.0);
    if (a_scaled) {
        printf("\nMatrix A * 2.0:\n");
        for (int i = 0; i < a_scaled->rows; i++) {
            for (int j = 0; j < a_scaled->cols; j++) {
                printf("%.4f ", matrix_get(a_scaled, i, j));
            }
            printf("\n");
        }
        matrix_free(a_scaled);
    }

    // Clean up
    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

// Function to test activation functions
void test_activation_functions() {
    printf("\n===== Testing Activation Functions =====\n");

    // Test values
    double test_values[] = { -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 };
    int num_values = sizeof(test_values) / sizeof(test_values[0]);

    printf("Input\tReLU\tSigmoid\tTanh\n");
    for (int i = 0; i < num_values; i++) {
        double x = test_values[i];
        double relu = RELu_function(x);
        double sigmoid = Sigmoid_function(x);
        double tanh_val = Tanh_function(x);

        printf("%.1f\t%.4f\t%.4f\t%.4f\n", x, relu, sigmoid, tanh_val);
    }

    // Test derivatives
    printf("\nInput\tReLU'\tSigmoid'\tTanh'\n");
    for (int i = 0; i < num_values; i++) {
        double x = test_values[i];

        // For relu derivative
        double relu_deriv = RELu_derivative_function(x);

        // For sigmoid, we need to calculate sigmoid first
        double sigmoid = Sigmoid_function(x);
        double sigmoid_deriv = Sigmoid_derivative_function(sigmoid);

        // For tanh, we need to calculate tanh first
        double tanh_val = Tanh_function(x);
        double tanh_deriv = Tanh_derivative_function(tanh_val);

        printf("%.1f\t%.4f\t%.4f\t%.4f\n", x, relu_deriv, sigmoid_deriv, tanh_deriv);
    }

    // In test_activation_functions, add this test
    double x = 0.5;
    double sigmoid_val = Sigmoid_function(x);
    double sigmoid_deriv = sigmoid_val * (1 - sigmoid_val);
    printf("Manual check: sigmoid(%.1f)=%.4f, sigmoid'(%.1f)=%.4f\n",
        x, sigmoid_val, x, sigmoid_deriv);
}

// Function to test a single neuron
void test_single_neuron() {
    printf("\n===== Testing Single Neuron =====\n");

    // Create a neuron with 2 inputs and sigmoid activation
    neuron* n = neuron_create(2, Sigmoid);
    if (!n) {
        fprintf(stderr, "Neuron creation failed\n");
        return;
    }

    // Print initial weights and bias
    printf("Initial weights: [%.4f, %.4f]\n", n->weights->data[0], n->weights->data[1]);
    printf("Initial bias: %.4f\n", n->bias);

    // Test activation with sample input
    Matrix* input = matrix_create(1, 2);
    if (!input) {
        fprintf(stderr, "Failed to create input matrix\n");
        neuron_free(n);
        return;
    }

    matrix_set(input, 0, 0, 0.5);
    matrix_set(input, 0, 1, -0.5);

    double output = neuron_activation(input, n);
    printf("Input: [0.5, -0.5]\nOutput: %.4f\n", output);

    // Test backward pass
    printf("\nTesting backward pass with output gradient 1.0...\n");
    Matrix* input_gradients = neuron_backward(1.0, n, 0.1);
    if (input_gradients) {
        printf("Input gradients: [%.4f, %.4f]\n",
            matrix_get(input_gradients, 0, 0),
            matrix_get(input_gradients, 0, 1));
        printf("Updated weights: [%.4f, %.4f]\n", n->weights->data[0], n->weights->data[1]);
        printf("Updated bias: %.4f\n", n->bias);
        matrix_free(input_gradients);
    }

    // Clean up
    matrix_free(input);
    neuron_free(n);
}

// Function to test a single layer
void test_single_layer() {
    printf("\n===== Testing Single Layer =====\n");

    // Create a layer with 3 neurons, each with 2 inputs
    layer* l = layer_create(3, 2, Sigmoid);
    if (!l) {
        fprintf(stderr, "Layer creation failed\n");
        return;
    }

    // Print layer structure
    printf("Layer created with %d neurons, each with %d inputs\n",
        l->neuronAmount, l->neurons[0]->weights->cols);

    // Test forward pass
    Matrix* input = matrix_create(1, 2);
    if (!input) {
        fprintf(stderr, "Failed to create input matrix\n");
        layer_free(l);
        return;
    }

    matrix_set(input, 0, 0, 0.5);
    matrix_set(input, 0, 1, -0.5);

    Matrix* output = layer_forward(l, input);
    if (output) {
        printf("Layer output for input [0.5, -0.5]:\n[");
        for (int i = 0; i < output->cols; i++) {
            printf("%.4f", matrix_get(output, 0, i));
            if (i < output->cols - 1) printf(", ");
        }
        printf("]\n");

        // Test backward pass
        printf("\nTesting backward pass...\n");
        Matrix* gradients = matrix_create(1, 3);
        if (gradients) {
            // Set some gradients for the output
            for (int i = 0; i < gradients->cols; i++) {
                matrix_set(gradients, 0, i, 1.0);
            }

            Matrix* input_gradients = layer_backward(l, gradients, 0.1);
            if (input_gradients) {
                printf("Input gradients: [%.4f, %.4f]\n",
                    matrix_get(input_gradients, 0, 0),
                    matrix_get(input_gradients, 0, 1));
                matrix_free(input_gradients);
            }
            matrix_free(gradients);
        }

        matrix_free(output);
    }

    // Clean up
    matrix_free(input);
    layer_free(l);
}

// Function to test neural network
void test_neural_network() {
    printf("\n===== Testing Neural Network =====\n");

    // Create a simple XOR network
    int layers[] = { 4, 1 };           // Hidden layer with 4 neurons, output layer with 1 neuron
    ActivationType activations[] = { Tanh, Sigmoid };

    network* net = network_create(2, layers, 2, activations, 0.5);
    if (!net) {
        fprintf(stderr, "Network creation failed\n");
        return;
    }

    printf("Network created with %d layers\n", net->layerAmount);

    // Create XOR dataset
    Matrix* inputs;
    Matrix* outputs;
    create_xor_dataset(&inputs, &outputs, 4);

    if (!inputs || !outputs) {
        fprintf(stderr, "Failed to create XOR dataset\n");
        network_free(net);
        return;
    }

    // Train the network
    printf("\nTraining network on XOR problem for 10000 epochs...\n");
    int epochs = 100000;
    print_network_weights(net);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;

        for (int i = 0; i < 4; i++) {
            // Get a single training example
            Matrix* single_input = get_row(inputs, i);
            Matrix* single_output = get_row(outputs, i);

            // Train on this example
            double error = train(net, single_input, single_output);
            total_error += error;
        }

        // Print progress every 1000 epochs
        if ((epoch + 1) % 1000 == 0) {
            printf("Epoch %d: Average error = %.6f\n", epoch + 1, total_error / 4);
        }
    }

    print_network_weights(net);
    // Test the trained network
    printf("\nTesting trained network on XOR problem:\n");
    printf("Input\t\tTarget\tPrediction\n");

    for (int i = 0; i < 4; i++) {
        Matrix* single_input = get_row(inputs, i);
        Matrix* prediction = forwardPropagation(net, single_input);

        if (prediction) {
            printf("[%.1f, %.1f]\t%.1f\t%.4f\n",
                matrix_get(inputs, i, 0),
                matrix_get(inputs, i, 1),
                matrix_get(outputs, i, 0),
                matrix_get(prediction, 0, 0));

            matrix_free(prediction);
        }
    }

    // Clean up
    matrix_free(inputs);
    matrix_free(outputs);
    network_free(net);
}

// Function to test creating a network through the empty constructor
void test_network_empty_constructor() {
    printf("\n===== Testing Network Empty Constructor =====\n");

    network* net = network_create_empty();
    if (!net) {
        fprintf(stderr, "Network empty creation failed\n");
        return;
    }

    printf("Empty network created\n");

    // Add layers manually
    if (add_layer(net, 4, Sigmoid, 2)) {
        printf("Added first layer: 2 inputs -> 4 neurons (SIGMOID)\n");
    }
    else {
        fprintf(stderr, "Failed to add first layer\n");
        network_free(net);
        return;
    }

    if (add_layer(net, 1, Sigmoid, 0)) {
        printf("Added second layer: 4 inputs -> 1 neuron (SIGMOID)\n");
    }
    else {
        fprintf(stderr, "Failed to add second layer\n");
        network_free(net);
        return;
    }

    // Set learning rate
    net->learnningRate = 0.1;
    printf("Set learning rate to 0.1\n");

    // Verify network structure
    printf("Network has %d layers\n", net->layerAmount);
    printf("Layer sizes: [%d, %d]\n", net->layersSize[0], net->layersSize[1]);

    // Clean up
    network_free(net);
}

int main() {
    printf("===== Neural Network Library Test Program =====\n");

    // Test matrix operations
    test_matrix_operations2();

    // Test activation functions
    test_activation_functions();

    // Test neuron functionality
    test_single_neuron();

    // Test layer functionality
    test_single_layer();

    // Test network empty constructor and add_layer
    test_network_empty_constructor();

    // Test full neural network
    test_neural_network();

    printf("\n===== Tests Completed =====\n");
    return 0;
}