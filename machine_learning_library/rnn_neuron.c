#include "neuron.h"
#include "rnn_neuron.h"

rnn_neuron* rnn_neuron_create(int weightslength, ActivationType func)
{
    if (weightslength <= 0) {
        fprintf(stderr, "Error: Invalid weights length %d or Invald in neuron_create\n", weightslength);
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
    
    rn->recurrent_weights = ((float )rand() / RAND_MAX) * 2.0 - 1.0;
    rn->grad_recurrent_weights = 0.0;
    rn->hidden_state = 0.0;
    rn->grad_hidden_state = 0.0;
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

float  rnn_neuron_activation(Tensor* input, rnn_neuron* rn)
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
    float  sum = 0.0;
    for (int i = 0; i < input->count; i++) {
        // Use proper tensor element access functions
        float  input_val = tensor_get_element_by_index(input, i);
        float  weight_val = tensor_get_element_by_index(rn->n->weights, i);
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

void rnn_neuron_backward(float  output_gradient, rnn_neuron* rn, Tensor* input_grads)
{
    if (!rn || !rn->input_history[rn->timestamp]) {
        fprintf(stderr, "Error: NULL neuron or input history in rnn_neuron_backward\n");
        return NULL;
    }

    rn->n->output = rn->hidden_state_history[rn->timestamp]; // its the output in this timestamp
    float  activation_derivative = rn->n->ActivationderivativeFunc(rn->n);

    // Chain rule - gradient flows through activation function
    float  pre_activation_gradient = output_gradient * activation_derivative;
    // For storing the gradient flowing back to the previous hidden state
    float  hidden_gradient = pre_activation_gradient * rn->recurrent_weights;

    for (int i = 0; i < rn->n->weights->count; i++) {
        float  original_weight = tensor_get_element_by_index(rn->n->weights, i);

        float  input_val = tensor_get_element_by_index(rn->input_history[rn->timestamp], i);

        input_grads->data[i] = pre_activation_gradient * original_weight;

        rn->n->grad_weights->data[i] += pre_activation_gradient * input_val;
    }

    rn->grad_recurrent_weights += pre_activation_gradient * rn->hidden_state_history[rn->timestamp];
    rn->grad_hidden_state += hidden_gradient;
    rn->n->grad_bias += pre_activation_gradient;
}

void rnn_neuron_update(rnn_neuron* rn, float lr)
{
    if (!rn || !rn->n) {
        fprintf(stderr, "Error: NULL rnn_neuron or inner neuron in rnn_neuron_update_weights\n");
        return;
    }

    neuron_update(rn->n, lr);
    rn->recurrent_weights += lr * rn->grad_recurrent_weights;
}

void rnn_neuron_zero_grad(rnn_neuron* rn)
{
    if (!rn || !rn->n) return;
    neuron_zero_grad(rn->n);
    rn->grad_recurrent_weights = 0.0f;
    rn->grad_hidden_state = 0.0f;
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