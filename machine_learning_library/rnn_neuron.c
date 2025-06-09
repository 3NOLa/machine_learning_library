#include "neuron.h"
#include "rnn_neuron.h"
#include "optimizers.h"

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
    rn->opt = (optimizer*)malloc(sizeof(optimizer));
    optimizer_set(rn->opt, SGD);
    if (!n || !rn->opt) {
        fprintf(stderr, "Error: Memory allocation failed for neuron or optimizer in rnn_neuron\n");
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
    float  sum = tensor_dot(rn->n->weights, input);

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

    // Calculate gradients for this neuron's parameters and inputs
    tensor_multiply_scalar_existing_more(
        (Tensor * []) {input_grads, rn->n->grad_weights},
        (Tensor * []) {rn->n->weights, rn->input_history[rn->timestamp]},
        (float[]) {pre_activation_gradient, pre_activation_gradient},
         2
    );

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
    rnn_neuron_opt_update(rn, rn->opt, lr);
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