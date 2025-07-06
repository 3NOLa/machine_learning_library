#include "dense_layer.h"
#include <stdlib.h>
#include "weights_initialization.h"
#include "optimizers.h"

dense_layer* dense_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc)
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
    L->init = initializer_xavier_normal(neuronDim,neuronAmount);
    L->opt = (optimizer*)malloc(sizeof(optimizer));
    optimizer_set(L->opt,SGD);

    L->weights = tensor_create(2, (int[]) { neuronAmount, neuronDim });
    L->init->tensor_init(L->weights, L->init);

    L->bias = tensor_create(1,(int []) {neuronAmount});
    L->init->tensor_init(L->bias, L->init);

    L->output = NULL;
    L->input = NULL;

    L->grad_weights = tensor_zero_create(2, (int[]) { neuronAmount, neuronDim });
    L->grad_bias = tensor_zero_create(1,(int []) {neuronAmount});
    L->input_grad = NULL;

    L->init->tensor_init(L->weights,L->init);
    L->init->tensor_init(L->bias,L->init);

    return L;
}

void layer_set_activtion(dense_layer* l, ActivationType Activationfunc)
{
    if(!l) {
        fprintf(stderr, "Error: NULL dense_layer in set_layer_activtion\n");
        return NULL;
    }

    l->Activationenum = Activationfunc;
}

void dense_layer_forward(dense_layer* l, Tensor* input) {
    if (!l || !input || input->dims < 1) {
        fprintf(stderr, "Error: Invalid input to dense_layer_forward\n");
        return;
    }

    if (!l->output) {
        int* new_shape = (int*)malloc(sizeof(int) * input->dims);
        if (!new_shape) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            return;
        }

        memcpy(new_shape, input->shape, sizeof(int) * input->dims);
        new_shape[input->dims - 1] = l->weights->shape[1];

        l->output = tensor_zero_create(input->dims, new_shape);

        free(new_shape); 
    }


    if (!l->input && !l->input_grad){
        l->input = tensor_create(input->dims,input->shape);
        l->input_grad = tensor_zero_create(input->dims,input->shape);
    }    
    tensor_copy(l->input,input);
    
    tensor_mmul(input,l->weights,l->output,false);
    tensor_brodcast_inplace(l->output,l->bias);
}

void dense_layer_backward(dense_layer* l, Tensor* output_gradients) {
    if (!l || !output_gradients) {
        fprintf(stderr, "Error: NULL dense_layer or gradients\n");
        return;
    }

    int output_dim = l->weights->shape[1];
    int input_dim = l->weights->shape[0];

    int batch = output_gradients->shape[0];
    int out_dim_check = output_gradients->shape[output_gradients->dims - 1];
    if (out_dim_check != output_dim) {
        fprintf(stderr, "Error: Gradient size mismatch %d , %d(2D and 3d case)\n",output_dim,out_dim_check);
        return;
    }

    tensor_mmul(output_gradients,l->input,l->grad_weights,true);
    tensor_mmul(output_gradients,l->weights,l->input_grad,true);
    
}

void dense_layer_update(dense_layer* l, float learning_rate) {
    l->opt->tensor_update(l->weights, l->grad_weights, learning_rate, &(l->opt->args));
    l->opt->tensor_update(l->bias, l->grad_bias, learning_rate, &(l->opt->args));
}

void dense_layer_set_optimizer(dense_layer* layer, OptimizerType type) {
    optimizer_set(layer->opt, type);
}

void dense_layer_zero_grad(dense_layer* l){
    if (!l) return;
    tensor_fill(l->grad_weights, 0.0f);
    tensor_fill(l->grad_bias, 0.0f);
    tensor_fill(l->input_grad, 0.0f);
}

void dense_layer_opt_init(dense_layer* dl, Initializer* init, initializerType type)
{
    if (!init) {
        switch (type) {
        case RandomNormal:
            init =  initializer_random_normal(0, 1);
            break;
        case RandomUniform:
            init = initializer_random_uniform(-1, 1);
            break;
        case XavierNormal:
            init = initializer_xavier_normal(dl->weights->shape[1], dl->neuronAmount);
            break;
        case XavierUniform:
            init = initializer_xavier_uniform(dl->weights->shape[1], dl->neuronAmount);
            break;
        case HeNormal:
            init = initializer_he_normal(dl->weights->shape[1]);
            break;
        case HeUniform:
            init = initializer_he_uniform(dl->weights->shape[1]);
            break;
        case LeCunNormal:
            init = initializer_lecun_normal(dl->weights->shape[1]);
            break;
        case LeCunUniform:
            init = initializer_lecun_uniform(dl->weights->shape[1]);
            break;
        //case Orthogonal:
         //   init = initializer_orthogonal(f1, i1, i2);
        //case Sparse:
            //init = initializer_sparse(i1, i2);
        default:
            fprintf(stderr, "Error: not a valid type or not implmeneted yet in dense_layer_opt_init\n");
            return; 
        }
    }

    init->tensor_init(dl->weights, init);
    init->tensor_init(dl->bias, init);
}

void layer_free(dense_layer* l)
{
    if (l) {
        if (l->bias) tensor_free(l->bias);
        if (l->weights) tensor_free(l->weights);
        if (l->grad_bias) tensor_free(l->grad_bias);
        if (l->grad_weights) tensor_free(l->grad_weights);
        if (l->input) tensor_free(l->input);
        if (l->input_grad) tensor_free(l->input_grad);
        if (l->output) tensor_free(l->output);
        
        free(l);
    }
}

int save_dense_layer_model(const FILE* wfp, const FILE* cfp, dense_layer* dl) {
    fprintf(cfp, "Layer Type = dense layer\n");
    fprintf(cfp, "neurons amount = %d\n", dl->neuronAmount);
    fprintf(cfp, "Activation type = %d\n", dl->Activationenum);
    fprintf(cfp, "Layer input dim = %d\n", dl->weights->dims);
    fprintf(cfp, "Layer shape = ");
    for (int i = 0; i < dl->weights->dims; i++) {
        fprintf(cfp, "%d, ", dl->weights->shape[i]);
    }
    fprintf(cfp, "\n");

    for (int i = 0; i < dl->neuronAmount; i++){
        fwrite(dl->weights->data, sizeof(float), dl->weights->count, wfp);
        fwrite(dl->bias->data, sizeof(float), dl->bias->count, wfp);
    }

    return 1;
}

int load_dense_layer_weights_model(const FILE* wfp, dense_layer* dl) {
    for (int i = 0; i < dl->neuronAmount; i++) {
        fprintf(stderr, "before weight: %f\n", dl->weights->data[0]);
        fprintf(stderr,"bytes read: %d\t", fread(dl->weights->data, sizeof(float), dl->weights->count, wfp));
        fprintf(stderr,"bytes read: %d\t", fread(&dl->bias, sizeof(float), 1, wfp));
        fprintf(stderr, "new weight: %f\n", dl->weights->data[0]);
    }
    fprintf(stderr, "\n");

    return 1;
}