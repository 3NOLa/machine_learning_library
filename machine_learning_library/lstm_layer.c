#include "dense_layer.h"
#include "rnn_layer.h"
#include "lstm_layer.h"


lstm_layer* lstm_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc)
{
    if (neuronAmount <= 0 || neuronDim <= 0) {
        fprintf(stderr, "Error: Invalid dimensions in lstm_layer_create - neurons: %d, dimension: %d\n",
            neuronAmount, neuronDim);
        return NULL;
    }

    lstm_layer* ll = (lstm_layer*)malloc(sizeof(lstm_layer));
    if (!ll) {
        fprintf(stderr, "Error: Memory allocation failed for lstm_layer_create\n");
        return NULL;
    }

    ll->Activationenum = Activationfunc;
    ll->neuronAmount = neuronAmount;
    ll->sequence_length = 0;

    ll->output = tensor_zero_create(1, (int[]) { neuronAmount });
    if (!ll->output) {
        fprintf(stderr, "Error: Failed to create output for lstm_layer_create\n");
        free(ll);
        return NULL;
    }

    ll->neurons = (lstm_neuron**)malloc(sizeof(lstm_neuron*) * neuronAmount);
    if (!ll->neurons) {
        fprintf(stderr, "Error: Memory allocation failed for neurons array in lstm_layer_create\n");
        free(ll);
        return NULL;
    }

    for (int i = 0; i < neuronAmount; i++) {
        ll->neurons[i] = lstm_neuron_create(neuronDim, Activationfunc);
        if (!ll->neurons[i]) {
            fprintf(stderr, "Error: Failed to create neuron %d in lstm_layer_create\n", i);
            for (int j = 0; j < i; j++) {
                lstm_neuron_free(ll->neurons[j]);
            }
            free(ll->neurons);
            free(ll);
            return NULL;
        }
    }

    return ll;
}

Tensor* lstm_layer_forward(lstm_layer* ll, Tensor* input)
{
    if (!ll || !input) {
        fprintf(stderr, "Error: NULL dense_layer or input in layer_forward\n");
        return NULL;
    }

    Tensor* output = tensor_create(1, (int[]) { ll->neuronAmount });
    if (!output) {
        fprintf(stderr, "Error: Failed to create output tensor in layer_forward\n");
        return NULL;
    }

    for (int i = 0; i < ll->neuronAmount; i++) {
        ll->neurons[i]->timestamp = ll->sequence_length;
        float  activation = lstm_neuron_activation(input, ll->neurons[i]);

        tensor_set(output, (int[]) { i }, activation);
    }

    if (ll->output)
        free(ll->output);
    ll->output = tensor_create(output->dims, output->shape);
    tensor_copy(ll->output, output);

    ll->sequence_length++;

    return output;
}

Tensor* lstm_layer_backward(lstm_layer* ll, Tensor* output_gradients)
{
    if (!ll || !output_gradients) {
        fprintf(stderr, "Error: NULL dense_layer or gradients in lstm_layer_backward\n");
        return NULL;
    }

    if (ll->sequence_length <= 0) {
        fprintf(stderr, "Error: No sequence processed yet in lstm_layer_backward\n");
        return NULL;
    }

    Tensor* accumulated_input_gradients = NULL;

    for (int t = ll->sequence_length - 1; t >= 0; t--) {

        Tensor* timestep_input_gradients = tensor_create(ll->neurons[0]->f_g->n->weights->dims, ll->neurons[0]->f_g->n->weights->shape);
        if (!timestep_input_gradients) {
            fprintf(stderr, "Error: Failed to create input gradients in lstm_neuron_backward\n");
            return NULL;
        };

        for (int i = 0; i < ll->neuronAmount; i++) {
            ll->neurons[i]->timestamp = t;

            float  output_gradient = tensor_get_element(output_gradients, (int[]) { 0, i });

            lstm_neuron_backward(output_gradient, ll->neurons[i], timestep_input_gradients);
        }

        // Accumulate gradients across timesteps
        if (accumulated_input_gradients == NULL) {
            accumulated_input_gradients = timestep_input_gradients;
        }
        else {
            // Add this timestep's gradients to the accumulated gradients
            for (int j = 0; j < timestep_input_gradients->count; j++) {
                float  current = tensor_get_element_by_index(accumulated_input_gradients, j);
                float  to_add = tensor_get_element_by_index(timestep_input_gradients, j);
                tensor_set_by_index(accumulated_input_gradients, j, current + to_add);
            }
            tensor_free(timestep_input_gradients);
        }
    }

    return accumulated_input_gradients;
}

void lstm_layer_update(lstm_layer* ll, float lr)
{
    if (!ll || ll->neuronAmount <= 0) {
        fprintf(stderr, "Error: NULL or empty rnn_layer in rnn_layer_update\n");
        return;
    }

    for (int i = 0; i < ll->neuronAmount; i++) {
        if (ll->neurons[i]) {
            lstm_neuron_update(ll->neurons[i], lr);
        }
    }
}

void lstm_layer_zero_grad(lstm_layer* ll)
{
    if (!ll) return;
    for (int i = 0; i < ll->neuronAmount; ++i)
        lstm_neuron_zero_grad(ll->neurons[i]);
}

void lstm_layer_reset_state(lstm_layer* ll)
{
    rnn_layer* rl = (rnn_layer*)ll;
    rnn_layer_reset_state(rl);
}

void lstm_layer_free(lstm_layer* ll)
{
    if (ll) {
        if (ll->neurons) {
            for (int i = 0; i < ll->neuronAmount; i++) {
                if (ll->neurons[i]) {
                    lstm_neuron_free(ll->neurons[i]);
                }
            }
            free(ll->neurons);
        }
        if (ll->output)
            tensor_free(ll->output);
        free(ll);
    }
}
