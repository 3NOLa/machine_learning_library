#include "dense_layer.h"
#include "rnn_layer.h"


rnn_layer* rnn_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc)
{
    if (neuronAmount <= 0 || neuronDim <= 0) {
        fprintf(stderr, "Error: Invalid dimensions in rnn_layer_create - neurons: %d, dimension: %d\n",
            neuronAmount, neuronDim);
        return NULL;
    }

    rnn_layer* rl = (rnn_layer*)malloc(sizeof(rnn_layer));
    if (!rl) {
        fprintf(stderr, "Error: Memory allocation failed for rnn_layer\n");
        return NULL;
    }

    rl->Activationenum = Activationfunc;
    rl->neuronAmount = neuronAmount;
    rl->sequence_length = 0;

    rl->output = tensor_zero_create(1, (int[]) { neuronAmount });
    if (!rl->output) {
        fprintf(stderr, "Error: Failed to create output for rnn_layer\n");
        free(rl);
        return NULL;
    }

    rl->neurons = (rnn_neuron**)malloc(sizeof(rnn_neuron*) * neuronAmount);
    if (!rl->neurons) {
        fprintf(stderr, "Error: Memory allocation failed for neurons array in rnn_layer\n");
        free(rl);
        return NULL;
    }

    for (int i = 0; i < neuronAmount; i++) {
        rl->neurons[i] = rnn_neuron_create(neuronDim, Activationfunc);
        if (!rl->neurons[i]) {
            fprintf(stderr, "Error: Failed to create neuron %d in rnn_layer\n", i);
            for (int j = 0; j < i; j++) {
                rnn_neuron_free(rl->neurons[j]);
            }
            free(rl->neurons);
            free(rl);
            return NULL;
        }
    }

    return rl;
}

Tensor* rnn_layer_forward(rnn_layer* rl, Tensor* input)
{
    if (!rl || !input) {
        fprintf(stderr, "Error: NULL dense_layer or input in layer_forward\n");
        return NULL;
    }

    // Create a 1D tensor for output
    Tensor* output = tensor_create(1, (int[]) { rl->neuronAmount });
    if (!output) {
        fprintf(stderr, "Error: Failed to create output tensor in layer_forward\n");
        return NULL;
    }

    for (int i = 0; i < rl->neuronAmount; i++) {
        //giving the neuron hidden state the output of the layer that is the t-1 output
        rl->neurons[i]->hidden_state = tensor_get_element(rl->output, (int[]) {i});
        //set neuron to the right timestamp
        rl->neurons[i]->timestamp = rl->sequence_length;
        float  activation = rnn_neuron_activation(input, rl->neurons[i]);

        tensor_set(output, (int[]) { i }, activation);
    }
    
    if (rl->output)
        free(rl->output);
    rl->output = tensor_create(output->dims, output->shape);
    tensor_copy(rl->output, output);

    rl->sequence_length++;

    return output;
}
 
Tensor* rnn_layer_backward(rnn_layer* rl, Tensor* output_gradients)
{
    if (!rl || !output_gradients) {
        fprintf(stderr, "Error: NULL dense_layer or gradients in rnn_layer_backward\n");
        return NULL;
    }

    if (rl->sequence_length <= 0) {
        fprintf(stderr, "Error: No sequence processed yet in rnn_layer_backward\n");
        return NULL;
    }

    // We need to create a tensor to accumulate input gradients
    // Assuming first neuron's weights shape tells us the input dimension
    Tensor* accumulated_input_gradients = NULL;

    // Start from the last timestep and work backwards (BPTT)
    for (int t = rl->sequence_length - 1; t >= 0; t--) {
        // For each timestep, we need gradients from all neurons
        Tensor* timestep_input_gradients = tensor_zero_create(
            rl->neurons[0]->n->weights->dims,
            rl->neurons[0]->n->weights->shape
        );

        for (int i = 0; i < rl->neuronAmount; i++) {
            rl->neurons[i]->timestamp = t;

            float  output_gradient = tensor_get_element(output_gradients, (int[]) {  i });

            rnn_neuron_backward(output_gradient, rl->neurons[i], timestep_input_gradients);
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

void rnn_layer_update(rnn_layer* rl, float lr)
{
    if (!rl || rl->neuronAmount <= 0) {
        fprintf(stderr, "Error: NULL or empty rnn_layer in rnn_layer_update\n");
        return;
    }

    for (int i = 0; i < rl->neuronAmount; i++) {
        if (rl->neurons[i]) {
            rnn_neuron_update(rl->neurons[i], lr);
        }
    }
}

void rnn_layer_zero_grad(rnn_layer* rl)
{
    if (!rl) return;
    for (int i = 0; i < rl->neuronAmount; ++i)
        rnn_neuron_zero_grad(rl->neurons[i]);
}

void rnn_layer_reset_state(rnn_layer* rl)
{
    if (!rl || !rl->neurons) return;
    for (int i = 0; i < rl->neuronAmount; i++) {
        rl->neurons[i]->hidden_state = 0.0;
        rl->neurons[i]->timestamp = 0;
    }
    rl->sequence_length = 0;
}

void rnn_layer_free(rnn_layer* rl)
{
    if (rl) {
        if (rl->neurons) {
            for (int i = 0; i < rl->neuronAmount; i++) {
                if (rl->neurons[i]) {
                    rnn_neuron_free(rl->neurons[i]);
                }
            }
            free(rl->neurons);
        }
        if (rl->output)
            tensor_free(rl->output);
        free(rl);
    }
}