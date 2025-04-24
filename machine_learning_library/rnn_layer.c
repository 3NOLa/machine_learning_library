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

    rl->output = tensor_zero_create(2, (int[]) { 1, neuronAmount });
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
        rl->neurons[i] = rnn_neuron_create(neuronDim, Activationfunc,rl->neuronAmount);
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

Tensor* rnn_layer_forward(rnn_layer* rl, Tensor* input,int t)
{
    if (!rl || !input) {
        fprintf(stderr, "Error: NULL dense_layer or input in layer_forward\n");
        return NULL;
    }

    // Create a 1D tensor for output
    Tensor* output = tensor_create(2, (int[]) { 1, rl->neuronAmount });
    if (!output) {
        fprintf(stderr, "Error: Failed to create output tensor in layer_forward\n");
        return NULL;
    }

    for (int i = 0; i < rl->neuronAmount; i++) {
        //giving the neuron hidden state the output of the layer that is the t-1 output
        rl->neurons[i]->hidden_state = tensor_get_element(rl->output, (int[]) {0,i});
        double activation = rnn_neuron_activation(input, rl->neurons[i],t);

        tensor_set(output, (int[]) { 0, i }, activation);
    }
    
    if (rl->output)
        free(rl->output);
    rl->output = tensor_create(output->dims, output->shape);
    tensor_copy(rl->output, output);

    rl->sequence_length++;

    return output;
}
 
Tensor* rnn_layer_backward(rnn_layer* rl, Tensor* output_gradients, double learning_rate)
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
        Tensor* timestep_input_gradients = NULL;

        for (int i = 0; i < rl->neuronAmount; i++) {
            // Get the gradient for this neuron's output
            double output_gradient = tensor_get_element(output_gradients, (int[]) { 0, i });

            // Backpropagate through this neuron
            Tensor* neuron_input_gradients = rnn_neuron_backward(output_gradient, rl->neurons[i], learning_rate, t);

            if (!neuron_input_gradients) {
                fprintf(stderr, "Error: Failed to get input gradients from neuron %d\n", i);
                continue;
            }

            // Initialize or accumulate gradients
            if (timestep_input_gradients == NULL) {
                timestep_input_gradients = tensor_create(neuron_input_gradients->dims, neuron_input_gradients->shape);
                tensor_copy(timestep_input_gradients, neuron_input_gradients);
            }
            else {
                // Add this neuron's gradients to the accumulated gradients
                for (int j = 0; j < neuron_input_gradients->count; j++) {
                    double current = tensor_get_element_by_index(timestep_input_gradients, j);
                    double to_add = tensor_get_element_by_index(neuron_input_gradients, j);
                    tensor_set_by_index(timestep_input_gradients, j, current + to_add);
                }
            }

            tensor_free(neuron_input_gradients);
        }

        // Accumulate gradients across timesteps
        if (accumulated_input_gradients == NULL) {
            accumulated_input_gradients = timestep_input_gradients;
        }
        else {
            // Add this timestep's gradients to the accumulated gradients
            for (int j = 0; j < timestep_input_gradients->count; j++) {
                double current = tensor_get_element_by_index(accumulated_input_gradients, j);
                double to_add = tensor_get_element_by_index(timestep_input_gradients, j);
                tensor_set_by_index(accumulated_input_gradients, j, current + to_add);
            }
            tensor_free(timestep_input_gradients);
        }
    }

    return accumulated_input_gradients;
}


void rnn_layer_reset_state(rnn_layer* rl)
{
    for (int i = 0; i < rl->neuronAmount; i++) {
        //tensor_zero(rl->neurons[i]->hidden_state);
        //rl->neurons[i]->timestep_count = 0;
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