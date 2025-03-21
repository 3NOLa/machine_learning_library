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

Matrix* layer_forward(layer* l, Matrix* input)
{
    if (!l || !input) {
        fprintf(stderr, "Error: NULL layer or input in layer_forward\n");
        return NULL;
    }

    Matrix* output = matrix_create(1, l->neuronAmount);
    if (!output) {
        fprintf(stderr, "Error: Failed to create output matrix in layer_forward\n");
        return NULL;
    }

    for (int i = 0; i < l->neuronAmount; i++) {
        double activation = neuron_activation(get_row(input,i), l->neurons[i],i);
        matrix_set(output, 0, i, activation);
    }

    return output;
}

Matrix* layer_backward(layer* l, Matrix* input_gradients, double learning_rate)
{
    if (!l || !input_gradients) {
        fprintf(stderr, "Error: NULL layer or gradients in layer_backward\n");
        return NULL;
    }

    if (input_gradients->cols != l->neuronAmount) {
        fprintf(stderr, "Error: Gradient size mismatch in layer_backward - got: %d, expected: %d\n",
            input_gradients->cols, l->neuronAmount);
        return NULL;
    }

    if (l->neuronAmount <= 0 || !l->neurons[0]) {
        fprintf(stderr, "Error: Layer has no neurons in layer_backward\n");
        return NULL;
    }

    // Output gradients with respect to this layer's inputs
    Matrix* output_gradients = matrix_zero_create(1, l->neurons[0]->weights->cols);
    if (!output_gradients) {
        fprintf(stderr, "Error: Failed to create output gradients in layer_backward\n");
        return NULL;
    }

    // For each neuron in the layer
    for (int i = 0; i < l->neuronAmount; i++) {
        // Get this neuron's portion of the gradient
        double neuron_gradient = matrix_get(input_gradients, 0, i);

        // Compute gradients for this neuron's weights and bias
        // Also get gradients with respect to inputs
        Matrix* neuron_input_gradients = neuron_backward(neuron_gradient, l->neurons[i], learning_rate);
        if (!neuron_input_gradients) {
            fprintf(stderr, "Error: Failed to compute neuron gradients in layer_backward\n");
            matrix_free(output_gradients);
            return NULL;
        }

        // Accumulate input gradients from this neuron
        for (int j = 0; j < output_gradients->cols; j++) {
            double current = matrix_get(output_gradients, 0, j);
            double to_add = matrix_get(neuron_input_gradients, 0, j);
            matrix_set(output_gradients, 0, j, current + to_add);
        }

        // Free the temporary gradients
        matrix_free(neuron_input_gradients);
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
        free(l);
    }
}