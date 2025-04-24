#include "neuron.h"
#include "rnn_neuron.h"

rnn_neuron* rnn_neuron_create(int weightslength, ActivationType func, int layer_amount)
{
    if (weightslength <= 0 || layer_amount <= 0) {
        fprintf(stderr, "Error: Invalid weights length %d or Invalid layer_amount length %d in neuron_create\n", weightslength, layer_amount);
        return NULL;
    }

    rnn_neuron* rn = (rnn_neuron*)malloc(sizeof(rnn_neuron));
    if (!rn) {
        fprintf(stderr, "Error: Memory allocation failed for rnn_neuron\n");
        return NULL;
    }

    neuron* n = neuron_create(weightslength, func);
    if (!n) {
        fprintf(stderr, "Error: Memory allocation failed for neuron in rnn_neuron\n");
        return NULL;
    }
    rn->n = n;
    
    rn->recurrent_weights = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    rn->hidden_state = 0.0;
    rn->timestamp = 0;

    for (int i = 0; i < MAX_TIMESTEPS; i++) {
        rn->input_history[i] = NULL;
        rn->hidden_state_history[i] = 0.0;
    }

    return rn;
}

void rnn_neuron_set_ActivationType(rnn_neuron* rn, ActivationType Activation)
{
    neuron_set_ActivationType(rn->n, Activation);
}

double rnn_neuron_activation(Tensor* input, rnn_neuron* rn)
{
    if (!input || !rn) {
        fprintf(stderr, "Error: NULL input or neuron in neuron_activation\n");
        return 0.0;
    }

    if (input->count != rn->n->weights->count) {
        fprintf(stderr, "Error: Size mismatch in neuron_activation - input count: %d, weights count: %d\n",
            input->count, rn->n->weights->count);
        return 0.0;
    }

    if (rn->n->input) {
        tensor_free(rn->n->input);
    }
    // Create a new tensor with the same dimensions and shape as input
    rn->n->input = tensor_create(input->dims, input->shape);
    rn->input_history[rn->timestamp] = tensor_create(input->dims, input->shape);
    if (!rn->n->input || !rn->input_history[rn->timestamp]) {
        fprintf(stderr, "Error: Failed to create input copy in rnn_neuron_activation\n");
        return 0.0;
    }

    if (!tensor_copy(rn->n->input, input) || !tensor_copy(rn->input_history[rn->timestamp],input)) {
        fprintf(stderr, "Error: Failed to copy input in rnn_neuron_activation\n");
        return 0.0;
    }

    // Calculate weighted sum using tensor operations
    double sum = 0.0;
    for (int i = 0; i < input->count; i++) {
        // Use proper tensor element access functions
        double input_val = tensor_get_element_by_index(input, i);
        double weight_val = tensor_get_element_by_index(rn->n->weights, i);
        sum += input_val * weight_val;
    }

    sum += rn->hidden_state * rn->recurrent_weights;
    sum += rn->n->bias;

    rn->n->pre_activation = sum;
    rn->n->output = rn->n->ActivationFunc(sum);
    
    rn->hidden_state = rn->n->output;
    rn->hidden_state_history[rn->timestamp] = rn->hidden_state;

    return rn->hidden_state;
}

Tensor* rnn_neuron_backward(double output_gradient, rnn_neuron* rn, double learning_rate)
{
    if (!rn || !rn->input_history[rn->timestamp]) {
        fprintf(stderr, "Error: NULL neuron or input history in rnn_neuron_backward\n");
        return NULL;
    }

    // Derivative of activation function w.r.t. its input
    rn->n->output = rn->hidden_state_history[rn->timestamp];
    double activation_derivative = rn->n->ActivationderivativeFunc(rn->n);

    // Chain rule - gradient flows through activation function
    double pre_activation_gradient = output_gradient * activation_derivative;

    // Create gradients for inputs with the same shape as weights
    Tensor* input_gradients = tensor_create(rn->n->weights->dims, rn->n->weights->shape);
    if (!input_gradients) {
        fprintf(stderr, "Error: Failed to create input gradients in rnn_neuron_backward\n");
        return NULL;
    }

    // For storing the gradient flowing back to the previous hidden state
    double hidden_gradient = pre_activation_gradient * rn->recurrent_weights;

    // Calculate gradients for this neuron's parameters and inputs
    for (int i = 0; i < rn->n->weights->count; i++) {
        // Get original weight using tensor access
        double original_weight = tensor_get_element_by_index(rn->n->weights, i);

        // Get input value using tensor access (from history at this timestamp)
        double input_val = tensor_get_element_by_index(rn->input_history[rn->timestamp], i);

        // Calculate weight gradient
        double weight_gradient = pre_activation_gradient * input_val;

        // Store input gradient using original weight
        tensor_set_by_index(input_gradients, i, pre_activation_gradient * original_weight);

        // Update weight
        tensor_set_by_index(rn->n->weights, i, original_weight + weight_gradient * learning_rate);
    }

    // Update recurrent weight
    double recurrent_gradient = pre_activation_gradient * rn->hidden_state_history[rn->timestamp];
    rn->recurrent_weights += recurrent_gradient * learning_rate;

    // Update bias
    rn->n->bias += pre_activation_gradient * learning_rate;

    // Return the gradient to propagate back to previous layer
    return input_gradients;
}

void rnn_neuron_free(rnn_neuron* rn)
{
    if (rn) {
        if (rn->n) {
            neuron_free(rn->n);
        }
        for (int i = 0; i < MAX_TIMESTEPS; i++) {
            if (rn->input_history[i]) {
                tensor_free(rn->input_history[i]);
            }
        }

        free(rn);
    }
}