#include "Tensor_Header.h"

bool tensor_add_row(Tensor* t)
{
    int row_size = 1;
    for (int i = 1; i < t->dims; i++)
        row_size *= t->shape[i];

    int new_size = t->count + row_size;
    float* new_data = (float*)realloc(t->data, sizeof(float)* new_size);
    if(!new_data) {
        fprintf(stderr, "Error: Memory allocation failed for tensor data in tensor_add_row\n");
        return false;
    }

    t->data = new_data;

    for (int i = t->count; i < new_size; i++) {
        t->data[i] = 0.0;
    }

    t->shape[0] += 1;

    t->count = new_size;

    return true;
}


void transpose_blocked(const float* src, float* dst, int rows, int cols) {
    for (int i = 0; i < rows; i += 8) {
        for (int j = 0; j < cols; j += 8) {
            int max_i = (i + 8 > rows) ? rows : i + 8;
            int max_j = (j + 8 > cols) ? cols : j + 8;

            for (int bi = i; bi < max_i; bi += 4) {
                for (int bj = j; bj < max_j; bj += 4) {
                    if (bi + 3 < rows && bj + 3 < cols) {
                        // Load 4 rows of 4 floats each
                        __m128 row0 = _mm_loadu_ps(&src[(bi + 0) * cols + bj]);
                        __m128 row1 = _mm_loadu_ps(&src[(bi + 1) * cols + bj]);
                        __m128 row2 = _mm_loadu_ps(&src[(bi + 2) * cols + bj]);
                        __m128 row3 = _mm_loadu_ps(&src[(bi + 3) * cols + bj]);

                        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

                        // Store into transposed locations
                        _mm_storeu_ps(&dst[(bj + 0) * rows + bi], row0);
                        _mm_storeu_ps(&dst[(bj + 1) * rows + bi], row1);
                        _mm_storeu_ps(&dst[(bj + 2) * rows + bi], row2);
                        _mm_storeu_ps(&dst[(bj + 3) * rows + bi], row3);
                    } else {
                        // Fallback for remaining elements
                        for (int ii = bi; ii < max_i && ii < rows; ++ii) {
                            for (int jj = bj; jj < max_j && jj < cols; ++jj) {
                                dst[jj * rows + ii] = src[ii * cols + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

Tensor* tensor_transpose(Tensor* t) {
    int* result_shape = t->shape;
    result_shape[t->dims - 1] ^= result_shape[t->dims - 2];
    result_shape[t->dims - 2] ^= result_shape[t->dims - 1];
    result_shape[t->dims - 1] ^= result_shape[t->dims - 2];

    Tensor* result = tensor_create(t->dims, result_shape);
    if (!result) return NULL;

    if(t->dims == 2)
        transpose_blocked(t->data, result->data, t->shape[0], t->shape[1]);
    else if (t->dims == 3) {
        for(int i=0;i<t->shape[0];i++)
            transpose_blocked(&t->data[t->strides[0]], &result->data[result->strides[0]], t->shape[1], t->shape[2]);
    }
    return result;
}

void tensor_transpose_inplace(Tensor* t) {
    float* tmp = malloc(sizeof(float) * t->count);
    if (!tmp) return;

    if (t->dims == 2)
        transpose_blocked(t->data, tmp, t->shape[0], t->shape[1]);
    else if (t->dims == 3) {
        for (int i = 0; i < t->shape[0]; i++)
            transpose_blocked(&t->data[t->strides[0]], &tmp[t->strides[0]], t->shape[1], t->shape[2]);
    }

    t->shape[t->dims - 1] ^= t->shape[t->dims - 2];
    t->shape[t->dims - 2] ^= t->shape[t->dims - 1];
    t->shape[t->dims - 1] ^= t->shape[t->dims - 2];

    memcpy(t->data, tmp, sizeof(float) * t->count);
    free(tmp);
}

Tensor* tensor_reshape(Tensor* t, int dims, int* shape) {
    if (!t || !shape) return NULL;

    // Calculate total elements in new shape
    int new_count = 1;
    for (int i = 0; i < dims; i++) {
        new_count *= shape[i];
    }

    // Check if reshape is valid
    if (new_count != t->count) {
        fprintf(stderr, "Error: Cannot reshape tensor - element count mismatch\n");
        return NULL;
    }

    Tensor* result = tensor_create(dims, shape);
    if (!result) return NULL;

    // Copy data
    memcpy(result->data, t->data, sizeof(float ) * t->count);

    return result;
}

Tensor* tensor_flatten(Tensor* t) {
    if (!t) return NULL;

    int shape[1] = { t->count };
    return tensor_reshape(t, 1, shape);
}

Tensor* tensor_slice_range(Tensor* t, int start, int end)
{
    if (!t || start < 0 || end > t->shape[0] || start >= end) {
        fprintf(stderr, "Error: Invalid parameters in tensor_slice_range start: %d, end %d\n",start,end);
        return NULL;
    }

    int outer_dim = end - start;
    int inner_count = t->count / t->shape[0];

    // Create shape for the sliced tensor
    int* new_shape = (int*)malloc(sizeof(int) * t->dims);
    if (!new_shape) {
        fprintf(stderr, "Error: Memory allocation failed in tensor_slice_range\n");
        return NULL;
    }
    new_shape[0] = outer_dim;
    for (int i = 1; i < t->dims; i++) {
        new_shape[i] = t->shape[i];
    }

    Tensor* result = tensor_create(t->dims, new_shape);
    free(new_shape);
    if (!result) return NULL;

    // Copy the slice
    int offset = start * inner_count;
    memcpy(result->data, t->data + offset, sizeof(float) * inner_count * outer_dim);

    return result;
}

void tensor_squeeze(Tensor* t) {
    int* new_shape = malloc(sizeof(int) * t->dims);  

    if(!new_shape)
    {
        fprintf(stderr, "Error: Memory allocation failed in tensor_squeeze\n");
        return;
    }

    int new_ndim = 0;
    for (int i = 0; i < t->dims; ++i) {
        if (t->shape[i] != 1)
            new_shape[new_ndim++] = t->shape[i];
    }

    free(t->shape);
    t->shape = realloc(new_shape, sizeof(int) * new_ndim);
    t->dims = new_ndim;
}


Tensor* tensor_slice_dim(Tensor* t, int start, int end,int dim)
{
    if (!t || start < 0 || end > t->shape[dim] || start >= end) {
        fprintf(stderr, "Error: Invalid parameters in tensor_slice_range\n");
        return NULL;
    }

    int outer_dim = end - start;
    int inner_count = t->count / t->shape[dim];

    // Create shape for the sliced tensor
    int* new_shape = (int*)malloc(sizeof(int) * t->dims);
    if (!new_shape) {
        fprintf(stderr, "Error: Memory allocation failed in tensor_slice_range\n");
        return NULL;
    }

    new_shape[dim] = outer_dim;
    for (int i = 0; i < t->dims; i++) {
        if(i != dim)
            new_shape[i] = t->shape[i];
    }

    Tensor* result = tensor_create(t->dims, new_shape);
    free(new_shape);
    if (!result) return NULL;

    // Copy the slice
    int offset = start * inner_count;
    memcpy(result->data, t->data + offset, sizeof(float) * inner_count * outer_dim);

    return result;
}

Tensor* tensor_get_row(Tensor* t, int row) {
    if (!t || t->dims < 2 || row < 0 || row >= t->shape[0]) {
        fprintf(stderr, "Error: Invalid parameters in tensor_get_row\n");
        return NULL;
    }

    // Create a tensor for the row
    int* new_shape = (int*)malloc(sizeof(int) * (t->dims - 1));
    if (!new_shape) {
        fprintf(stderr, "Error: Memory allocation failed in tensor_get_row\n");
        return NULL;
    }

    // Copy remaining dimensions
    for (int i = 1; i < t->dims; i++) {
        new_shape[i - 1] = t->shape[i];
    }
    Tensor* row_tensor = tensor_create(t->dims - 1, new_shape);
    free(new_shape);

    if (!row_tensor) return NULL;

    // Copy data for this row
    int* indices = (int*)malloc(sizeof(int) * t->dims);
    indices[0] = row;
    int* new_indices = (int*)malloc(sizeof(int) * (t->dims - 1));
    for (int i = 0; i < row_tensor->count; i++) {
        // Convert flat index to multi-dimensional indices for the row tensor
        int temp = i;
        for (int j = t->dims - 2; j >= 0; j--) {
            new_indices[j] = temp % row_tensor->shape[j];
            temp /= row_tensor->shape[j];
        }

        // Map to original tensor
        for (int j = 1; j < t->dims; j++) {
            indices[j] = new_indices[j - 1];
        }

        // Get value and set in row tensor
        float  value = tensor_get_element(t, indices);
        tensor_set_by_index(row_tensor, i, value);
    }

    return row_tensor;
}
