#include "Tensor_Header.h"

void tensor_set_by_index(Tensor* t, int index, float  value) {
    if (!t || index < 0 || index >= t->count) {
        fprintf(stderr, "Error: Index out of bounds in tensor_set_by_index\n");
        return;
    }

    t->data[index] = value;
}


int tensor_get_index(Tensor* t, int* indices) {
    if (!t) {
        fprintf(stderr, "Error: NULL tensor in tensor_get_index\n");
        return -1;  // Return -1 to indicate error
    }
    if (!indices) {
        fprintf(stderr, "Error: NULL indices in tensor_get_index\n");
        return -1;
    }

    int index = 0;
    for (int i = 0; i < t->dims; i++) {
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            fprintf(stderr, "Error: Index %d out of bounds for dimension %d (size %d) in tensor_get_index\n",
                indices[i], i, t->shape[i]);
            return -1;
        }
        index += indices[i] * t->strides[i];
    }

    return index;
}


float  tensor_get_element_by_index(Tensor* t, int index)
{
    return t->data[index];
}

float  tensor_get_element(Tensor* t, int* indices) {
    int index = tensor_get_index(t, indices);
    if (index == -1) return 0.0; // Error case

    return tensor_get_element_by_index(t, index);
}

void tensor_set(Tensor* t, int* indices, float  value) {
    if (!t) {
        fprintf(stderr, "Error: NULL matrix in tensor_set\n");
        return;
    }
    t->data[tensor_get_index(t,indices)] = value;
}

void tensor_set_bt_index(Tensor* t, int index, float  value) {
    if (!t) {
        fprintf(stderr, "Error: NULL matrix in tensor_set\n");
        return;
    }
    t->data[index] = value;
}