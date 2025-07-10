#pragma once
#include "export.h"


typedef struct Tensor Tensor;

// Access functions
EXPORT int tensor_get_index(Tensor* t, int* indices);
EXPORT float  tensor_get_element(Tensor* t, int* indices);
EXPORT float  tensor_get_element_by_index(Tensor* t, int index);
EXPORT void tensor_set(Tensor* t, int* indices, float  value);
EXPORT void tensor_set_by_index(Tensor* t, int index, float  value);