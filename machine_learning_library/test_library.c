#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "network.h"
#include "csv_parser.h"


void print_network_weights(network* net) {
    printf("Network weights:\n");
    for (int l = 0; l < net->layerAmount; l++) {
        printf("Layer %d:\n", l);
        for (int n = 0; n < net->layersSize[l]; n++) {
            neuron* neuron = net->layers[l]->neurons[n];
            printf("  Neuron %d: bias=%.4f weights=[", n, neuron->bias);
            tensor_print(neuron->weights);
            printf("]\n");
        }
    }
}

// Function to create a simple XOR dataset
void create_xor_dataset(Tensor** inputs, Tensor** outputs, int num_samples) {
    // Create inputs: [0,0], [0,1], [1,0], [1,1]
    int input_shape[2] = { num_samples, 2 };  // num_samples x 2 tensor
    int output_shape[2] = { num_samples, 1 }; // num_samples x 1 tensor

    *inputs = tensor_create(2, input_shape);
    *outputs = tensor_create(2, output_shape);

    if (!*inputs || !*outputs) {
        fprintf(stderr, "Failed to create dataset tensors\n");
        return;
    }

    // XOR truth table - using proper tensor indexing
    int indices[2];

    // Sample 0: [0,0] -> 0
    indices[0] = 0; indices[1] = 0;
    tensor_set(*inputs, indices, 0.0);
    indices[0] = 0; indices[1] = 1;
    tensor_set(*inputs, indices, 0.0);
    indices[0] = 0; indices[1] = 0;
    tensor_set(*outputs, indices, 0.0);

    // Sample 1: [0,1] -> 1
    indices[0] = 1; indices[1] = 0;
    tensor_set(*inputs, indices, 0.0);
    indices[0] = 1; indices[1] = 1;
    tensor_set(*inputs, indices, 1.0);
    indices[0] = 1; indices[1] = 0;
    tensor_set(*outputs, indices, 1.0);

    // Sample 2: [1,0] -> 1
    indices[0] = 2; indices[1] = 0;
    tensor_set(*inputs, indices, 1.0);
    indices[0] = 2; indices[1] = 1;
    tensor_set(*inputs, indices, 0.0);
    indices[0] = 2; indices[1] = 0;
    tensor_set(*outputs, indices, 1.0);

    // Sample 3: [1,1] -> 0
    indices[0] = 3; indices[1] = 0;
    tensor_set(*inputs, indices, 1.0);
    indices[0] = 3; indices[1] = 1;
    tensor_set(*inputs, indices, 1.0);
    indices[0] = 3; indices[1] = 0;
    tensor_set(*outputs, indices, 0.0);
}

// Function to test matrix operations
void test_matrix_operations2() {
    printf("\n===== Testing Matrix Operations =====\n");

    // Test matrix creation
    int shape1[2] = { 2,3 };
    int shape2[2] = { 3,2 };

    Tensor* a = tensor_random_create(2, shape1);
    printf("Matrix A (2x3, random):\n");
    for (int i = 0; i < a->count; i++)
        printf("%f, ", a->data[i]);
    tensor_print(a);

    Tensor* b = tensor_random_create(2, shape2);
    Tensor* c = tensor_identity_create(3);

    if (!a || !b || !c) {
        fprintf(stderr, "Matrix creation failed\n");
        return;
    }

    printf("\nMatrix B (3x2, random):\n");
    tensor_print(b);

    printf("\nMatrix C (3x3, identity):\n");
    tensor_print(c);

    // Test tensor multiplication
    Tensor* ab = tensor_multiply(a, b);
    if (ab) {
        printf("\nMatrix A * B (2x2):\n");
        tensor_print(ab);
    }
    else {
        printf("Matrix multiplication failed\n");
    }

    // Test tensor scalar operations
    Tensor* a_scaled = tensor_multiply_scalar(a,2.0);
    if (a_scaled) {
        printf("\nMatrix A * 2.0:\n");
        tensor_print(a_scaled);
        tensor_free(a_scaled);
    }

    // Clean up
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

// Function to test activation functions
void test_activation_functions() {
    printf("\n===== Testing Activation Functions =====\n");

    // Test values
    double test_values[] = { -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 };
    int num_values = sizeof(test_values) / sizeof(test_values[0]);

    printf("Input\tReLU\tSigmoid\tTanh\tlinear\tGelu\tleaky_relu\n");
    for (int i = 0; i < num_values; i++) {
        double x = test_values[i];
        double relu = RELu_function(x);
        double sigmoid = Sigmoid_function(x);
        double tanh_val = Tanh_function(x);
        double linear = linear_function(x);
        double gelu = gelu_function(x);
        double leaky_relu = leaky_RELu_function(x);

        printf("%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", x, relu, sigmoid, tanh_val,linear,gelu,leaky_relu);
    }

    // Test derivatives
    neuron* n = neuron_create(1,RELU);
    printf("Input\tReLU'\tSigmoid'\tTanh'\tlinear'\tGelu'\tleaky_relu'\n");
    for (int i = 0; i < num_values; i++) {
        n->pre_activation = test_values[i];

        n->output = RELu_function(n->pre_activation);
        double relu_derivative = RELu_derivative_function(n);

        n->output = Sigmoid_function(n->pre_activation);
        double sigmoid_derivative = Sigmoid_derivative_function(n);

        n->output = Tanh_function(n->pre_activation);
        double tanh_derivative = Tanh_derivative_function(n);

        n->output = linear_function(n->pre_activation);
        double linear_derivative = linear_derivative_function(n);

        n->output = gelu_function(n->pre_activation);
        double gelu_derivative = gelu_derivative_function(n);

        double leaky_relu = leaky_RELu_function(n->pre_activation);
        double leaky_relu_derivative = leaky_RELu_derivative_function(n);

        printf("%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", test_values[i], relu_derivative, sigmoid_derivative, tanh_derivative, linear_derivative, gelu_derivative, leaky_relu_derivative);
    }
    neuron_free(n);

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
    neuron* n = neuron_create(2, SIGMOID);
    if (!n) {
        fprintf(stderr, "Neuron creation failed\n");
        return;
    }

    // Print initial weights and bias
    printf("Initial weights: [%.4f, %.4f]\n", n->weights->data[0], n->weights->data[1]);
    printf("Initial bias: %.4f\n", n->bias);

    // Test activation with sample input
    int shape[2] = {1,2};
    Tensor* input = tensor_create(2, shape);
    if (!input) {
        fprintf(stderr, "Failed to create input matrix\n");
        neuron_free(n);
        return;
    }

    int index1[2] = { 0,0 };
    int index2[2] = { 0,1 };
    tensor_set(input, index1, 0.5);
    tensor_set(input, index2, -0.5);

    double output = neuron_activation(input, n);
    printf("Input: [0.5, -0.5]\nOutput: %.4f\n", output);

    // Test backward pass
    printf("\nTesting backward pass with output gradient 1.0...\n");
    Tensor* input_gradients = neuron_backward(1.0, n, 0.01);
    if (input_gradients) {
        printf("Input gradients: [%.4f, %.4f]\n",
            tensor_get_element_by_index(input_gradients, 0),
            tensor_get_element_by_index(input_gradients, 1));
        printf("Updated weights: [%.4f, %.4f]\n", n->weights->data[0], n->weights->data[1]);
        printf("Updated bias: %.4f\n", n->bias);
        tensor_free(input_gradients);
    }

    // Clean up
    tensor_free(input);
    neuron_free(n);
}

// Function to test a single layer
void test_single_layer() {
    printf("\n===== Testing Single Layer =====\n");

    // Create a layer with 3 neurons, each with 2 inputs
    layer* l = layer_create(3, 2, SIGMOID);
    if (!l) {
        fprintf(stderr, "Layer creation failed\n");
        return;
    }

    // Print layer structure
    printf("Layer created with %d neurons, each with %d inputs\n",
        l->neuronAmount, l->neurons[0]->weights->shape[1]);

    // Test forward pass
    int shape[2] = { 1, 2 };
    Tensor* input = tensor_create(2,shape);
    if (!input) {
        fprintf(stderr, "Failed to create input matrix\n");
        layer_free(l);
        return;
    }

    int index1[2] = { 0,0 };
    int index2[2] = { 0,1 };
    tensor_set(input, index1, 0.5);
    tensor_set(input, index2, -0.5);

    Tensor* output = layer_forward(l, input);
    if (output) {
        printf("Layer output for input [0.5, -0.5]:\n[");
        for (int i = 0; i < output->shape[1]; i++) {
            int index3[2] = { 0,i };
            printf("%.4lf", tensor_get_element(output, index3));
            if (i < output->shape[1] - 1) printf(", ");
        }
        printf("]\n");

        // Test backward pass
        printf("\nTesting backward pass...\n");
        int shape3[2] = { 1, 3 };
        Tensor* gradients = tensor_create(2,shape3);
        if (gradients) {
            // Set some gradients for the output
            for (int i = 0; i < gradients->shape[1]; i++) {
                int index4[2] = { 0,i };
                tensor_set(gradients, index4, 1.0);
            }

            Tensor* input_gradients = layer_backward(l, gradients, 0.1);
            if (input_gradients) {
                printf("Input gradients: [%.4f, %.4f]\n",
                    tensor_get_element(input_gradients, index1),
                    tensor_get_element(input_gradients, index2));
                tensor_free(input_gradients);
            }
            tensor_free(gradients);
        }

        tensor_free(output);
    }

    // Clean up
    tensor_free(input);
    layer_free(l);
}

// Function to test neural network
void test_neural_network() {
    printf("\n===== Testing Neural Network =====\n");

    // Create a simple XOR network
    int layers[] = { 4, 1 };           // Hidden layer with 4 neurons, output layer with 1 neuron
    int input_shape[] = { 1,2 };
    ActivationType activations[] = { GELU, LEAKY_RELU };

    network* net = network_create(2, layers, 2, input_shape, activations, 0.1,MAE);
    if (!net) {
        fprintf(stderr, "Network creation failed\n");
        return;
    }

    printf("Network created with %d layers\n", net->layerAmount);

    // Create XOR dataset
    Tensor* inputs;
    Tensor* outputs;
    create_xor_dataset(&inputs, &outputs, 4);

    if (!inputs || !outputs) {
        fprintf(stderr, "Failed to create XOR dataset\n");
        network_free(net);
        return;
    }

    // Train the network
    printf("\nTraining network on XOR problem for 10000 epochs...\n");
    int epochs = 1000;
    print_network_weights(net);
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;

        for (int i = 0; i < 4; i++) {
            // Get a single training example using tensor_slice instead of tensor_get_row
            int start_indices[2] = { i, 0 };
            int end_indices[2] = { i + 1, inputs->shape[1] };
            Tensor* single_input = tensor_get_row(inputs,i);
            Tensor* single_output = tensor_get_row(outputs,i);

            if (!single_input || !single_output) {
                fprintf(stderr, "Failed to slice training example %d\n", i);
                continue;
            }

            // Train on this example
            double error = train(net, single_input, single_output);
            total_error += error;

            // Free the sliced tensors
            tensor_free(single_input);
            tensor_free(single_output);
        }

        // Print progress every 1000 epochs
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch %d: Average error = %.6f\n", epoch + 1, total_error / 4);
        }
    }

    print_network_weights(net);

    // Test the trained network
    printf("\nTesting trained network on XOR problem:\n");
    printf("Input\t\tTarget\tPrediction\n");

    for (int i = 0; i < 4; i++) {
        // Get input example
        Tensor* single_input = tensor_get_row(inputs, i);
        if (!single_input) {
            fprintf(stderr, "Failed to slice test example %d\n", i);
            continue;
        }

        // Forward pass
        Tensor* prediction = forwardPropagation(net, single_input);

        if (prediction) {
            // Get the input values using proper tensor indexing
            int input_indices1[2] = { i, 0 };
            int input_indices2[2] = { i, 1 };
            int output_indices[2] = { i, 0 };
            int pred_indices[2] = { 0,0 }; // Assuming prediction is a 1D tensor

            printf("[%.1f, %.1f]\t%.1f\t%.4f\n",
                tensor_get_element(inputs, input_indices1),
                tensor_get_element(inputs, input_indices2),
                tensor_get_element(outputs, output_indices),
                tensor_get_element(prediction, pred_indices));

            tensor_free(prediction);
        }

        tensor_free(single_input);
    }

    // Clean up
    tensor_free(inputs);
    tensor_free(outputs);
    network_free(net);
}

void test_2d_input() {
    // Create a simple 2D pattern
    int input_shape[2] = { 3, 3 };
    Tensor* input = tensor_create(2, input_shape);

    // Set values for a simple pattern (e.g., a diagonal line)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int indices[2] = { i, j };
            if (i == j) {
                tensor_set(input, indices, 1.0);
            }
            else {
                tensor_set(input, indices, 0.0);
            }
        }
    }

    // Create a simple network
    int layerSizes[2] = { 4, 1 };  // Hidden layer with 4 neurons, output layer with 1 neuron
    ActivationType activations[2] = { SIGMOID, SIGMOID };

    // Initialize network with the input dimensions
    network* net = network_create(2, layerSizes, 2, input_shape, activations, 0.1, MSE);

    // Create a simple target (e.g., 1 for diagonal line pattern)
    int target_shape[2] = { 1, 1 };
    Tensor* target = tensor_create(2, target_shape);
    int target_idx[2] = { 0, 0 };
    tensor_set(target, target_idx, 1.0);

    // Forward pass and check output
    Tensor* output = forwardPropagation(net, input);
    printf("Output for diagonal pattern: %.4f\n", tensor_get_element_by_index(output, 0));

    // Train for a few iterations
    for (int i = 0; i < 1000; i++) {
        double error = train(net, input, target);
        if (i % 100 == 0) {
            printf("Iteration %d, Error: %.4f\n", i, error);
        }
    }

    // Test again after training
    output = forwardPropagation(net, input);
    printf("Output after training: %.4f\n", tensor_get_element_by_index(output, 0));

    // Clean up
    tensor_free(input);
    tensor_free(target);
    tensor_free(output);
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
    if (add_layer(net, 4, SIGMOID, 2)) {
        printf("Added first layer: 2 inputs -> 4 neurons (SIGMOID)\n");
    }
    else {
        fprintf(stderr, "Failed to add first layer\n");
        network_free(net);
        return;
    }

    if (add_layer(net, 1, SIGMOID, 0)) {
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

void test_csv_parser(char* filename)
{
    csv_handler* h = csv_handler_create(filename);
    read_file_to_tensor(h);
    print_csv_file_hand(h);
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

    test_2d_input();

    test_csv_parser("C:\\Users\\keyna\\Downloads\\annual-enterprise-survey-2023-financial-year-provisional.csv");

    printf("\n===== Tests Completed =====\n");
    return 0;
}