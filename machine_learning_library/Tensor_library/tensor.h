#pragma once
#include "export.h"

typedef struct HashMap HashMap; //forward declare for dfs

EXPORT typedef struct Tensor{
    int dims; 
    int *shape; //shaoes of the dims
    int *strides; //amount of bytes i need to get to next dim;
    int count; //amount of elemnts
    float *data;
    float *grad;

    struct Function* grad_func;
    bool is_leaf;
    bool requires_grad;

    char id[32]; // an id for every tensor based on the adress pointer of that tensor
} Tensor;

EXPORT typedef struct Function{
    void (*backward)(struct Function*, Tensor* );
    Tensor** inputs;
    int num_inputs;
}Function;

EXPORT Tensor* tensor_create(int dims, int* shape);
EXPORT Function* function_create(int num_inputs, Tensor** inputs, void (*backward)(struct Function*, Tensor* ));
EXPORT Tensor* tensor_create_flatten(int dims, int* shape,float* flatten, int count);
EXPORT Tensor* tensor_zero_create(int dims, int* shape);
EXPORT Tensor* tensor_random_create(int dims, int* shape);
EXPORT Tensor* tensor_identity_create(int row);

EXPORT void tensor_backward(Tensor* t);
EXPORT void tensor_dfs(Tensor* t, Tensor ***topo_sorted, int *count, HashMap *m); // its a pointer to an array of pointers to tensors

EXPORT void tensor_free(Tensor* t);
EXPORT void function_free(Function* f);
EXPORT float* tensor_free_data(Tensor* t);
EXPORT float* tensor_free_grad(Tensor* t);
EXPORT bool tensor_copy(Tensor* dest, Tensor* src);

EXPORT void tensor_fill(Tensor* t,float value);

// Print tensor
EXPORT void print_tensor_recursive(float* data, int* shape, int dims, int depth, int offset);
EXPORT void tensor_print(Tensor* t);

