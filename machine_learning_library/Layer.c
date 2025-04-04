#include "layer.h"

layer* layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc)
{
    if (neuronAmount <= 0 || neuronDim <= 0) {
        fprintf(stderr, "Error: Invalid dimensions in layer_create - neurons: %d, dimension: %d\n",
            neuronAmount, neuronDim);
        return NULL;
    }

    layer* L = (layer*)malloc(sizeof(layer));
    if (!L) {
        fprintf(stderr, "Error: Memory allocation failed for layer\n");
        return NULL;
    }

    L->Activationenum = Activationfunc;
    L->neuronAmount = neuronAmount;
    L->output = NULL;

    L->neurons = (neuron**)malloc(sizeof(neuron*) * neuronAmount);
    if (!L->neurons) {
        fprintf(stderr, "Error: Memory allocation failed for neurons array\n");
        free(L);
        return NULL;
    }

    // Create each neuron
    for (int i = 0; i < neuronAmount; i++) {
        L->neurons[i] = neuron_create(neuronDim, Activationfunc);
        if (!L->neurons[i]) {
            fprintf(stderr, "Error: Failed to create neuron %d\n", i);
            // Free previously created neurons
            for (int j = 0; j < i; j++) {
                neuron_free(L->neurons[j]);
            }
            free(L->neurons);
            free(L);
            return NULL;
        }
    }

    return L;
}

void layer_removeLastNeuron(layer* l)
{
    neuron* last = l->neurons[l->neuronAmount - 1];
    if (!last)return;
    neuron_free(last);

    l->neurons = (neuron**)realloc(l->neurons, sizeof(neuron*) * (--l->neuronAmount));
}

void layer_addNeuron(layer* l)
{
    neuron* newn = neuron_create(l->neurons[0]->weights->shape[1],l->Activationenum);
    l->neurons = (neuron**)realloc(l->neurons, sizeof(neuron*) * (l->neuronAmount+1));
    l->neurons[l->neuronAmount++] = newn;
}

void layer_set_neuronAmount(layer* l, int neuronAmount)
{
    if (l->neuronAmount == neuronAmount) return;
    else if(l->neuronAmount > neuronAmount)
    {
        for (int i = l->neuronAmount; i > neuronAmount; i--)
            layer_removeLastNeuron(l);
    }
    else
    {
        for (int i = neuronAmount; i < neuronAmount; i++)
            layer_addNeuron(l);
    }
    
}

void layer_set_activtion(layer* l, ActivationType Activationfunc)
{
    if(!l) {
        fprintf(stderr, "Error: NULL layer in set_layer_activtion\n");
        return NULL;
    }

    for (int i = 0; i < l->neuronAmount; i++)
    {
        neuron_set_ActivationType(l->neurons[i], Activationfunc);
    }
}

Tensor* layer_forward(layer* l, Tensor* input)
{
    if (!l || !input) {
        fprintf(stderr, "Error: NULL layer or input in layer_forward\n");
        return NULL;
    }

    // Create a 1D tensor for output
    int outShape[2] = {1, l->neuronAmount };
    Tensor* output = tensor_create(2, outShape);
    if (!output) {
        fprintf(stderr, "Error: Failed to create output tensor in layer_forward\n");
        return NULL;
    }

    for (int i = 0; i < l->neuronAmount; i++) {
        // Remove the third parameter from neuron_activation call
        double activation = neuron_activation(input, l->neurons[i]);

        // Using the proper tensor_set function
        int indices[2] = {0, i };
        tensor_set(output, indices, activation);
    }

    if (l->output)
        free(l->output);
    l->output = tensor_create(output->dims,output->shape);
    tensor_copy(l->output, output);

    return output;
}

Tensor* layer_backward(layer* l, Tensor* input_gradients, double learning_rate)
{
    if (!l || !input_gradients) {
        fprintf(stderr, "Error: NULL layer or gradients in layer_backward\n");
        return NULL;
    }

    if (input_gradients->count != l->neuronAmount) { // count because only one row
        fprintf(stderr, "Error: Gradient size mismatch in layer_backward - got: %d, expected: %d\n",
            input_gradients->count, l->neuronAmount);
        return NULL;
    }

    if (l->neuronAmount <= 0 || !l->neurons[0]) {
        fprintf(stderr, "Error: Layer has no neurons in layer_backward\n");
        return NULL;
    }

    // Output gradients with respect to this layer's inputs
    // Create a tensor with the same shape as neuron weights
    Tensor* output_gradients = tensor_zero_create(2, l->neurons[0]->weights->shape);
    if (!output_gradients) {
        fprintf(stderr, "Error: Failed to create output gradients in layer_backward\n");
        return NULL;
    }

    // For each neuron in the layer
    for (int i = 0; i < l->neuronAmount; i++) {
        // Get this neuron's portion of the gradient using proper tensor access
        int grad_indices[2] = {0, i };
        double neuron_gradient = tensor_get_element(input_gradients, grad_indices);

        // Compute gradients for this neuron's weights and bias
        // Also get gradients with respect to inputs
        Tensor* neuron_input_gradients = neuron_backward(neuron_gradient, l->neurons[i], learning_rate);
        if (!neuron_input_gradients) {
            fprintf(stderr, "Error: Failed to compute neuron gradients in layer_backward\n");
            tensor_free(output_gradients);
            return NULL;
        }

        // Accumulate input gradients from this neuron
        for (int j = 0; j < output_gradients->count; j++) {
            int out_indices[2] = {0, j };
            int in_indices[2] = {0, j };

            double current = tensor_get_element(output_gradients, out_indices);
            double to_add = tensor_get_element(neuron_input_gradients, in_indices);

            tensor_set(output_gradients, out_indices, current + to_add);
        }

        // Free the temporary gradients
        tensor_free(neuron_input_gradients);
    }

    return output_gradients;
}

void layer_free(layer* l)
{
    if (l) {
        if (l->neurons) {
            for (int i = 0; i < l->neuronAmount; i++) {
                if (l->neurons[i]) {
                    neuron_free(l->neurons[i]);
                }
            }
            free(l->neurons);
        }
        if (l->output)
            free(l->output);
        free(l);
    }
}