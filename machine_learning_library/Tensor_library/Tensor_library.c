#include "Tensor_Header.h"

Tensor* tensor_create(int dims, int* shape) {
    if (dims <= 0 || !shape) {
        fprintf(stderr, "Error: Invalid tensor dimensions %d\n", dims);
        return NULL;
    }

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        fprintf(stderr, "Error: Memory allocation failed for tensor\n");
        return NULL;
    }

    t->shape = (int*)malloc(sizeof(int) * dims);
    t->strides = (int*)malloc(sizeof(int) * dims);
    if (!t->shape || !t->strides) {
        fprintf(stderr, "Error: Memory allocation failed for tensor shape/strides\n");
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    t->count = 1;
    for (int i = 0; i < dims; i++) {
        t->shape[i] = shape[i];
        t->count *= shape[i];
    }

    // Compute strides (right to left)
    t->strides[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }

    t->data = (float*)malloc(sizeof(float) * t->count);
    t->grad = (float*)calloc( t->count, sizeof(float));
    if (!t->data || !t->grad) {
        fprintf(stderr, "Error: Memory allocation failed for tensor data\n");
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }
    snprintf(t->id, sizeof(t->id), "%p", (void*)t); // setting id

    t->dims = dims;
    t->is_leaf = true;
    t->requires_grad = true;
    t->grad_func = NULL;

    return t;
}

Function* function_create(int num_inputs, Tensor** inputs, void (*backward)(struct Function*, Tensor* grad_output)){

    Function* f = (Function*)malloc(sizeof(Function));
    if (!f) {
        fprintf(stderr, "Error: Memory allocation failed for Function function_create\n");
        return NULL;
    }

    f->inputs = (Tensor**)malloc(sizeof(Tensor*) * num_inputs);
    if (!f->inputs) {
        fprintf(stderr, "Error: Memory allocation failed for inputs function_create\n");
        free(f);
        return NULL;
    }

    for (int i = 0; i < num_inputs; i++){
        f->inputs[i] = inputs[i];
    }

    f->backward = backward;
    f->num_inputs = num_inputs;

    return f;
}

Tensor* tensor_create_flatten(int dims, int* shape, float* flatten, int count)
{
    if (dims <= 0 || !shape || !flatten) {
        fprintf(stderr, "Error: Invalid tensor dimensions %d in tensor_create_flatten\n", dims);
        return NULL;
    }

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        fprintf(stderr, "Error: Memory allocation failed for tensor\n");
        return NULL;
    }

    t->count = count;
    t->dims = dims;
    t->shape = (int*)malloc(sizeof(int) * dims);
    t->data = (float*)malloc(sizeof(float) * count);
    
    t->strides = (int*)malloc(sizeof(int) * dims);
    if (!t->shape || !t->strides || !t->data) {
        fprintf(stderr, "Error: Memory allocation failed for tensor shape/strides/data in tensor_create_flatten\n");
        free(t->shape);
        free(t->strides);
        free(t->data);
        free(t);
        return NULL;
    }

    memcpy(t->shape, shape, sizeof(int) * dims);
    memcpy(t->data, flatten, sizeof(float) * count);

    t->strides[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }

    return t;
}

Tensor* tensor_zero_create(int dims, int* shape) {
    Tensor* t = tensor_create(dims, shape);
    if (!t) return NULL;

    __m256 vf = _mm256_set1_ps(0.0);
    int i = 0;
    for (; i < t->count - 8; i += 8) {
        _mm256_storeu_ps(&t->data[i], vf);
    }

    for (; i < t->count; ++i) {
        t->data[i] = 0.0;
    }
    return t;
}

void tensor_fill(Tensor* t, float value) {
    if (!t || !t->data) return;

    __m256 vf = _mm256_set1_ps(value);
    int i = 0;
    for (; i < t->count - 8; i += 8) {
        _mm256_storeu_ps(&t->data[i], vf);
    }

    for (; i < t->count; ++i) {
        t->data[i] = value;
    }
}

Tensor* tensor_random_create(int dims, int* shape) {
    Tensor* t = tensor_create(dims, shape);
    if (!t) return NULL;

    srand(time(NULL));
    for (int i = 0; i < t->count; i++) {
        t->data[i] = ((float )rand() / RAND_MAX); // Range [-1, 1]
    }

    return t;
}

Tensor* tensor_identity_create(int size) {
    int shape[2] = { size, size };
    Tensor* t = tensor_zero_create(2, shape); // Create a 2D tensor
    if (!t) return NULL;

    // Set diagonal elements to 1
    for (int i = 0; i < size; i++) {
        int indices[2] = { i, i };
        tensor_set(t, indices, 1.0);
    }

    return t;
}

void tensor_dfs(Tensor* t, Tensor ***topo_sorted, int *count, HashMap *m){
    hashmap_put(m, t->id, *count);

    if (t->grad_func){
        for (int i = 0; i < t->grad_func->num_inputs; i++){
            if(!hashmap_containes(m, t->id)){
                tensor_dfs(t->grad_func->inputs[i], topo_sorted, count, m);
            }
        }
    }
    
    *topo_sorted = (Tensor**)realloc(*topo_sorted, sizeof(Tensor*) * (*count + 1));
    if (!*topo_sorted) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    (*topo_sorted)[(*count)++] = t;
}

void tensor_backward(Tensor* t){
    Tensor **topo_sorted = NULL;
    int count = 0;
    HashMap *map = hashmap_create(10);

    tensor_dfs(t, &topo_sorted, &count, map);

    array_add_scalar(topo_sorted[0]->grad, 1.0f, topo_sorted[0]->grad, topo_sorted[0]->count); // grad to become one;
    fprintf(stderr, "count : %d\n",count);
    
    for(int i = count - 1; i>=0; i--){
        if (!topo_sorted[i]->is_leaf){
            topo_sorted[i]->grad_func->backward(topo_sorted[i]->grad_func, topo_sorted[i]);
        }
    }

    hashmap_free(map);
    free(topo_sorted);
}

void tensor_free(Tensor* t) {
    if (!t) return;
    
    free(t->shape);
    free(t->strides);
    free(t->data);
    free(t->grad);
    if (t->grad_func) {
        function_free(t->grad_func);
    }
    free(t);
}

void function_free(Function* f) {
    if (!f) return;
    
    free(f->inputs);
    free(f);
}

float* tensor_free_data(Tensor* t){
    float *data = t->data;
    if (t) {
        if (t->grad) free(t->grad);
        if (t->shape) free(t->shape);
        if (t->strides) free(t->strides);
        free(t);
    }
    return data;
}

float* tensor_free_grad(Tensor* t){
    float *grad = t->grad;
    if (t) {
        if (t->data) free(t->data);
        if (t->shape) free(t->shape);
        if (t->strides) free(t->strides);
        free(t);
    }
    return grad;
}

bool tensor_copy(Tensor* dest,Tensor* src) {
    if (!src || !dest) return false;
    memcpy(dest->data, src->data, sizeof(float) * src->count);
    return true;
}

void print_tensor_recursive(float* data, int* shape, int dims, int depth, int offset) {
    if (depth == dims - 1) {
        // Last dimension: print flat array
        fprintf(stderr, "[");
        for (int i = 0; i < shape[depth]; i++) {
            fprintf(stderr, "%.4f", data[offset + i]);
            if (i < shape[depth] - 1) fprintf(stderr, ", ");
        }
        fprintf(stderr, "]");
    } else {
        fprintf(stderr, "[\n");
        int stride = 1;
        for (int i = depth + 1; i < dims; i++) {
            stride *= shape[i];
        }
        for (int i = 0; i < shape[depth]; i++) {
            for (int j = 0; j < depth; j++) fprintf(stderr, "  "); // indentation
            print_tensor_recursive(data, shape, dims, depth + 1, offset + i * stride);
            if (i < shape[depth] - 1) fprintf(stderr, ",\n");
        }
        fprintf(stderr, "\n");
        for (int j = 0; j < depth - 1; j++) fprintf(stderr, "  ");
        fprintf(stderr, "]");
    }
}

void tensor_print(Tensor* t) {
    print_tensor_recursive(t->data, t->shape, t->dims, 0, 0);
    fprintf(stderr, "\n");
}

void tensor_print_grad(Tensor* t) {
    fprintf(stderr, "\n");
    print_tensor_recursive(t->grad, t->shape, t->dims, 0, 0);
    fprintf(stderr, "\n");
}

