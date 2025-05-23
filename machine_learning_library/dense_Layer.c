#include "dense_layer.h"

dense_layer* layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc)
{
    if (neuronAmount <= 0 || neuronDim <= 0) {
        fprintf(stderr, "Error: Invalid dimensions in layer_create - neurons: %d, dimension: %d\n",
            neuronAmount, neuronDim);
        return NULL;
    }

    dense_layer* L = (dense_layer*)malloc(sizeof(dense_layer));
    if (!L) {
        fprintf(stderr, "Error: Memory allocation failed for dense_layer\n");
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

void layer_removeLastNeuron(dense_layer* l)
{
    neuron* last = l->neurons[l->neuronAmount - 1];
    if (!last)return;
    neuron_free(last);

    l->neurons = (neuron**)realloc(l->neurons, sizeof(neuron*) * (--l->neuronAmount));
}

void layer_addNeuron(dense_layer* l)
{
    neuron* newn = neuron_create(l->neurons[0]->weights->shape[1],l->Activationenum);
    l->neurons = (neuron**)realloc(l->neurons, sizeof(neuron*) * (l->neuronAmount+1));
    l->neurons[l->neuronAmount++] = newn;
}

void layer_set_neuronAmount(dense_layer* l, int neuronAmount)
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

void layer_set_activtion(dense_layer* l, ActivationType Activationfunc)
{
    if(!l) {
        fprintf(stderr, "Error: NULL dense_layer in set_layer_activtion\n");
        return NULL;
    }

    for (int i = 0; i < l->neuronAmount; i++)
    {
        neuron_set_ActivationType(l->neurons[i], Activationfunc);
    }
}

Tensor* layer_forward(dense_layer* l, Tensor* input)
{
    if (!l || !input) {
        fprintf(stderr, "Error: NULL dense_layer or input in layer_forward\n");
        return NULL;
    }

    // Create a 1D tensor for output
    Tensor* output = tensor_create(1, (int[]){ l->neuronAmount });
    if (!output) {
        fprintf(stderr, "Error: Failed to create output tensor in layer_forward\n");
        return NULL;
    }

    for (int i = 0; i < l->neuronAmount; i++) {
        float  activation = neuron_activation(input, l->neurons[i]);

        tensor_set(output, (int[]) {  i }, activation);
    }

    if (l->output)
        free(l->output);
    l->output = tensor_create(output->dims,output->shape);
    tensor_copy(l->output, output);

    return output;
}

Tensor* layer_backward(dense_layer* l, Tensor* input_gradients)
{
    if (!l || !input_gradients) {
        fprintf(stderr, "Error: NULL dense_layer or gradients in layer_backward\n");
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
    Tensor* output_gradients = tensor_zero_create(l->neurons[0]->weights->dims, l->neurons[0]->weights->shape);
    if (!output_gradients) {
        fprintf(stderr, "Error: Failed to create output gradients in layer_backward\n");
        return NULL;
    }

    for (int i = 0; i < l->neuronAmount; i++) {
        int grad_indices[1] = { i };
        float  neuron_gradient = tensor_get_element(input_gradients, grad_indices);
        neuron_backward(neuron_gradient, l->neurons[i], output_gradients);
    }

    return output_gradients;
}

void dense_layer_update(dense_layer* layer, float learning_rate) {
    for (int i = 0; i < layer->neuronAmount; i++) {
        neuron_update(layer->neurons[i], learning_rate);
    }
}

void dense_layer_zero_grad(dense_layer* layer)
{
    if (!layer) return;
    for (int i = 0; i < layer->neuronAmount; i++) {
        neuron_zero_grad(layer->neurons[i]);
    }
}

void layer_free(dense_layer* l)
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