#include "neuron.h"
#include "rnn_neuron.h"
#include "lstm_neuron.h"

lstm_neuron* lstm_neuron_create(int weightslength, ActivationType func, int layer_amount)
{
    if (weightslength <= 0 || layer_amount <= 0) {
        fprintf(stderr, "Error: Invalid weights length %d or Invalid layer_amount length %d in lstm_neuron_create\n", weightslength, layer_amount);
        return NULL;
    }

    lstm_neuron* ln = (lstm_neuron*)malloc(sizeof(lstm_neuron));
    if (!ln) {
        fprintf(stderr, "Error: Memory allocation failed for lstm_neuron_create\n");
        return NULL;
    }

    rnn_neuron* f_g = rnn_neuron_create(weightslength, SIGMOID, layer_amount);
    rnn_neuron* i_g_r = rnn_neuron_create(weightslength, SIGMOID, layer_amount);
    rnn_neuron* i_g_p = rnn_neuron_create(weightslength, TANH, layer_amount);
    rnn_neuron* o_g_r = rnn_neuron_create(weightslength, SIGMOID, layer_amount);
    if (!f_g || !i_g_r || !i_g_p || !o_g_r) {
        fprintf(stderr, "Error: Memory allocation failed for neuron in lstm_neuron_create\n");
        return NULL;
    }
    ln->f_g = f_g;
    ln->i_g_r = i_g_r;
    ln->i_g_p = i_g_p;
    ln->o_g_r = o_g_r;

    ln->short_memory = 0.0;
    ln->long_memory = 0.0;
    ln->timestamp = 0;

    for (int i = 0; i < MAX_TIMESTEPS; i++) {
        ln->input_history[i] = NULL;
        ln->long_memory_history[i] = 0.0;
        ln->short_memory_history[i] = 0.0;
    }

    return ln;
}

double lstm_neuron_activation(Tensor* input, lstm_neuron* ln)
{
    if (!input || !ln) {
        fprintf(stderr, "Error: NULL input or neuron in neuron_activation\n");
        return 0.0;
    }

    ln->input_history[ln->timestamp] = tensor_create(input->dims, input->shape);
    if (!ln->input_history[ln->timestamp]) {
        fprintf(stderr, "Error: Failed to create input copy in lstm_neuron_activation\n");
        return 0.0;
    }

    if (!tensor_copy(ln->input_history[ln->timestamp], input)) {
        fprintf(stderr, "Error: Failed to copy input in lstm_neuron_activation\n");
        return 0.0;
    }
    
    //forget gate
    ln->f_g->timestamp = ln->timestamp;
    ln->f_g->hidden_state = ln->short_memory;
    double f_g_sum = rnn_neuron_activation(input,ln->f_g);

    ln->long_memory *= f_g_sum;
    
    //input gate
    ln->i_g_r->timestamp = ln->timestamp;
    ln->i_g_r->hidden_state = ln->short_memory;
    double i_g_r_sum = rnn_neuron_activation(input, ln->i_g_r);

    ln->i_g_p->timestamp = ln->timestamp;
    ln->i_g_p->hidden_state = ln->short_memory;
    double i_g_p_sum = rnn_neuron_activation(input, ln->i_g_p);

    ln->long_memory += i_g_r_sum * i_g_p_sum;

    //output gate
    ln->o_g_r->timestamp = ln->timestamp;
    ln->o_g_r->hidden_state = ln->short_memory;
    double o_g_r_sum = rnn_neuron_activation(input, ln->o_g_r);

    double o_g_p_sum = Tanh_function(ln->long_memory);
    ln->long_memory_history[ln->timestamp] = ln->short_memory = o_g_r_sum * o_g_p_sum;

    return ln->short_memory_history[ln->timestamp] = ln->short_memory;
}

Tensor* lstm_neuron_backward(double derivative, lstm_neuron* ln, double learning_rate)
{
    if (!ln) {
        fprintf(stderr, "Error: NULL neuron or input history in lstm_neuron_backward\n");
        return NULL;
    }

    Tensor* input_gradients = tensor_create(ln->f_g->n->weights->dims, ln->f_g->n->weights->shape);
    if (!input_gradients) {
        fprintf(stderr, "Error: Failed to create input gradients in lstm_neuron_backward\n");
        return NULL;
    }

    double activation_derivative = Tanh_function(ln->long_memory_history[ln->timestamp]);
    double pre_activation_gradient = derivative * activation_derivative;
    ln->o_g_r->timestamp = ln->timestamp;
    Tensor* o_g_r_tensor = rnn_neuron_backward(pre_activation_gradient, ln->o_g_r, learning_rate);
    
    neuron n_value;
    neuron* n = &n_value;
    n->output = ln->long_memory_history[ln->timestamp];
    double long_term_derviatve = Tanh_derivative_function(n) * derivative * ln->o_g_r->hidden_state_history[ln->o_g_r->timestamp];//DIDNT ADD L / ct+1 * ft +1

    ln->f_g->timestamp = ln->timestamp;
    double f_g_derivative = long_term_derviatve * ln->long_memory_history[ln->timestamp - 1];
    Tensor* f_g_tensor = rnn_neuron_backward(f_g_derivative, ln->f_g, learning_rate);

    ln->i_g_r->timestamp = ln->timestamp;
    double i_g_r_derivative = long_term_derviatve * ln->i_g_p->hidden_state_history[ln->timestamp];
    Tensor* i_g_r_tensor = rnn_neuron_backward(derivative, ln->i_g_r, learning_rate);

    ln->i_g_p->timestamp = ln->timestamp;
    double i_g_p_derivative = long_term_derviatve * ln->i_g_r->hidden_state_history[ln->timestamp];
    Tensor* i_g_p_tensor = rnn_neuron_backward(i_g_p_derivative, ln->i_g_p, learning_rate);

    tensor_add_more_inplace(o_g_r_tensor, (Tensor* []) { f_g_tensor , i_g_r_tensor, i_g_p_tensor}, 3);
    tensor_free(f_g_tensor);
    tensor_free(i_g_r_tensor);
    tensor_free(i_g_p_tensor);

    return o_g_r_tensor;
}

void lstm_neuron_free(lstm_neuron* ln)
{
    if (ln) {
        if (ln->f_g) rnn_neuron_free(ln->f_g);
        if (ln->i_g_p) rnn_neuron_free(ln->i_g_p);
        if (ln->i_g_r) rnn_neuron_free(ln->i_g_r);
        if (ln->o_g_r) rnn_neuron_free(ln->o_g_r);

        for (int i = 0; i < MAX_TIMESTEPS; i++) {
            if (ln->input_history[i]) {
                tensor_free(ln->input_history[i]);
            }
        }

        free(ln);
    }
}