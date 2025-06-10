#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h> // For AVX/AVX2
#include <string.h>
#include <time.h>

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

    t->data = (float *)malloc(sizeof(float ) * t->count);
    if (!t->data) {
        fprintf(stderr, "Error: Memory allocation failed for tensor data\n");
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    t->dims = dims;
    return t;
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
    return t;
}

Tensor* tensor_random_create(int dims, int* shape) {
    Tensor* t = tensor_create(dims, shape);
    if (!t) return NULL;

    srand(time(NULL));
    for (int i = 0; i < t->count; i++) {
        t->data[i] = ((float )rand() / RAND_MAX) * 2.0 - 1.0; // Range [-1, 1]
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

int tensor_add_row(Tensor* t)
{
    int row_size = 1;
    for (int i = 1; i < t->dims; i++)
        row_size *= t->shape[i];

    int new_size = t->count + row_size;
    float* new_data = (float*)realloc(t->data, sizeof(float)* new_size);
    if(!new_data) {
        fprintf(stderr, "Error: Memory allocation failed for tensor data in tensor_add_row\n");
        return;
    }

    t->data = new_data;

    for (int i = t->count; i < new_size; i++) {
        t->data[i] = 0.0;
    }

    t->shape[0] += 1;

    t->count = new_size;

    return 1;
}

void tensor_free(Tensor* t) {
    if (t) {
        if (t->data) free(t->data);
        if (t->shape) free(t->shape);
        if (t->strides) free(t->strides);
        free(t);
    }
}

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

int tensor_copy(Tensor* dest,Tensor* src) {
    if (!src || !dest) return NULL;

    //Tensor* t = tensor_create(src->dims, src->shape);
    //if (!t) return NULL;

    memcpy(dest->data, src->data, sizeof(float ) * src->count);
    return 1;
    //return t;
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
        return NULL;
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

/*Tensor* tensor_get_col(Tensor* t, int col) {
    if (!t || t->dims < 2 || col < 0 || col >= t->shape[1]) {
        fprintf(stderr, "Error: Invalid parameters in tensor_get_col\n");
        return NULL;
    }

    // Create a tensor for the column
    int* new_shape = (int*)malloc(sizeof(int) * (t->dims - 1));
    if (!new_shape) {
        fprintf(stderr, "Error: Memory allocation failed in tensor_get_col\n");
        return NULL;
    }

    // First dimension is the same, skip the column dimension
    new_shape[0] = t->shape[0];
    for (int i = 2; i < t->dims; i++) {
        new_shape[i - 1] = t->shape[i];
    }

    Tensor* col_tensor = tensor_create(t->dims - 1, new_shape);
    free(new_shape);

    if (!col_tensor) return NULL;

    // Copy data for this column
    int indices[t->dims];
    indices[1] = col;

    int new_indices[t->dims - 1];
    for (int i = 0; i < col_tensor->count; i++) {
        // Convert flat index to multi-dimensional indices for the column tensor
        int temp = i;
        for (int j = t->dims - 2; j >= 0; j--) {
            new_indices[j] = temp % col_tensor->shape[j];
            temp /= col_tensor->shape[j];
        }

        // Map to original tensor
        indices[0] = new_indices[0];
        for (int j = 2; j < t->dims; j++) {
            indices[j] = new_indices[j - 1];
        }

        // Get value and set in column tensor
        float  value = tensor_get_element(t, indices);
        tensor_set_by_index(col_tensor, i, value);
    }

    return col_tensor;
}*/

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    // Check if dimensions match
    if (a->dims != b->dims) {
        fprintf(stderr, "Error: Tensor dimensions don't match for addition\n");
        return NULL;
    }

    // Check if shapes match
    for (int i = 0; i < a->dims; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes don't match for addition\n");
            return NULL;
        }
    }

    // Create result tensor
    Tensor* result = tensor_create(a->dims, a->shape);
    if (!result) return NULL;

    //this code use a special cpu instruion that copy 8 elemnts at once(parallelism)
    int i = 0;
    for (; i <= a->count - 8; i+=8) { 
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result->data[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < a->count; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

void tensor_add_inplace(Tensor* target, Tensor* other) {
    if (!target || !other) return;

    // Check if dimensions match
    if (target->dims != other->dims) {
        fprintf(stderr, "Error: Tensor dimensions don't match for in-place addition\n");
        return;
    }

    // Check if shapes match
    for (int i = 0; i < target->dims; i++) {
        if (target->shape[i] != other->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes don't match for in-place addition\n");
            return;
        }
    }

    // Add elements
    //this code use a special cpu instruion that copy 8 elemnts at once(parallelism)
    int i = 0;
    for (; i <= target->count - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&other->data[i]);
        __m256 vb = _mm256_loadu_ps(&target->data[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&target->data[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < target->count; i++) {
        target->data[i] += other->data[i];
    }

}

void tensor_add_more_inplace(Tensor* target, Tensor* others[],int amount) {

    for (int i = 0; i < amount; i++)
    {
        if (!target || !others[i]) return;

        // Check if dimensions match
        if (target->dims != others[i]->dims) {
            fprintf(stderr, "Error: Tensor dimensions don't match for in-place addition\n");
            return;
        }

        // Check if shapes match
        for (int i = 0; i < target->dims; i++) {
            if (target->shape[i] != others[i]->shape[i]) {
                fprintf(stderr, "Error: Tensor shapes don't match for in-place addition\n");
                return;
            }
        }
    }

    // Add elements
    for (int i = 0; i < amount; i++) {
        //this code use a special cpu instruion that copy 8 elemnts at once(parallelism)
        int j = 0;
        for (; j <= target->count - 8; j += 8) {
            __m256 va = _mm256_loadu_ps(&target->data[j]);
            __m256 vb = _mm256_loadu_ps(&others[i]->data[j]);
            __m256 vr = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&target->data[j], vr);
        }

        // must add it because the loop before stops 7 elemnts or less before the end
        for (; j < target->count; j++) {
            target->data[j] += others[i]->data[j];
        }
    }
}

Tensor* tensor_subtract(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    // Check if dimensions match
    if (a->dims != b->dims) {
        fprintf(stderr, "Error: Tensor dimensions don't match for subtraction\n");
        return NULL;
    }

    // Check if shapes match
    for (int i = 0; i < a->dims; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes don't match for subtraction\n");
            return NULL;
        }
    }

    // Create result tensor
    Tensor* result = tensor_create(a->dims, a->shape);
    if (!result) return NULL;

    int i = 0;
    for (; i <= a->count - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(&result->data[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < a->count; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

Tensor* tensor_multiply(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    // Check if dimensions match
    if (a->dims != b->dims) {
        fprintf(stderr, "Error: Tensor dimensions don't match for element-wise multiplication\n");
        return NULL;
    }

    // Check if shapes match
    for (int i = 0; i < a->dims; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes don't match for element-wise multiplication\n");
            return NULL;
        }
    }

    // Create result tensor
    Tensor* result = tensor_create(a->dims, a->shape);
    if (!result) return NULL;

    int i = 0;
    for (; i <= a->count - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result->data[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < a->count; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    return result;
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    // Check if dimensions match
    if (a->dims != b->dims) {
        fprintf(stderr, "Error: Tensor dimensions don't match for tensor_div\n");
        return NULL;
    }

    // Check if shapes match
    for (int i = 0; i < a->dims; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes don't match for tensor_div\n");
            return NULL;
        }
    }

    // Create result tensor
    Tensor* result = tensor_create(a->dims, a->shape);
    if (!result) return NULL;

    int i = 0;
    for (; i <= a->count - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        __m256 vr = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(&result->data[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < a->count; i++) {
        result->data[i] = a->data[i] / b->data[i];
    }

    return result;
}

float sum8(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);             // low 128
    __m128 vhigh = _mm256_extractf128_ps(v, 1);            // high 128
    __m128 sum128 = _mm_add_ps(vlow, vhigh); //{s0,s1,s2,s3}              // add low and high

    // now do 4-lane horizontal sum
    __m128 shuf = _mm_movehdup_ps(sum128);  // {s1, s1, s3, s3}
    __m128 sums = _mm_add_ps(sum128, shuf); // {s0+s1, 2s1, s2+s3, 2s3}
    shuf = _mm_movehl_ps(shuf, sums); // {s2+s3, 2s3, ?, ?}
    sums = _mm_add_ss(sums, shuf); // {s0+s1+s2+s3, 2s1 + 2s3,?,?}

    return _mm_cvtss_f32(sums);  // from register first num to a float
}

float tensor_dot(Tensor* a, Tensor* b) {
    if (!a || !b) return 0.0;

    // Check if dimensions match
    if (a->dims != b->dims) {
        fprintf(stderr, "Error: Tensor dimensions don't match for element-wise multiplication\n");
        return 0.0;
    }

    // Check if shapes match
    for (int i = 0; i < a->dims; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes don't match for element-wise multiplication\n");
            return 0.0;
        }
    }

    // Create result tensor
    float result = 0.0;

    int i = 0;
    for (; i <= a->count - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        result += sum8(vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < a->count; i++) {
        result += a->data[i] * b->data[i];
    }

    return result;
}

Tensor* tensor_mmul(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    // For now, we only implement matrix multiplication (2D tensors)
    if (a->dims != 2 || b->dims != 2) {
        fprintf(stderr, "Error: tensor_mmul currently only supports 2D tensors\n");
        return NULL;
    }

    // Check dimensions for matrix multiplication
    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication\n");
        return NULL;
    }

    // Create result tensor
    int result_shape[2] = { a->shape[0], b->shape[1] };
    Tensor* result = tensor_zero_create(2, result_shape);
    if (!result) return NULL;

    // Perform matrix multiplication
    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < b->shape[1]; j++) {
            float  sum = 0.0;
            for (int k = 0; k < a->shape[1]; k++) {
                int a_indices[2] = { i, k };
                int b_indices[2] = { k, j };
                sum += tensor_get_element(a, a_indices) * tensor_get_element(b, b_indices);
            }
            int result_indices[2] = { i, j };
            tensor_set(result, result_indices, sum);
        }
    }

    return result;
}

Tensor* tensor_add_scalar(Tensor* t, float  scalar) {
    if (!t) return NULL;

    Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;

    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i < t->count - 8; i+=8) {
        __m256 vt = _mm256_loadu_ps(&t->data[i]);
        __m256 vr = _mm256_add_ps(vt, vs);
        _mm256_storeu_ps(&result->data[i], vr);
    }

    for (; i < result->count; i++) {
        result->data[i] = t->data[i] + scalar;
    }

    return result;
}

Tensor* tensor_subtract_scalar(Tensor* t, float  scalar) {
    if (!t) return NULL;

    Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;

    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i < t->count - 8; i += 8) {
        __m256 vt = _mm256_loadu_ps(&t->data[i]);
        __m256 vr = _mm256_sub_ps(vt, vs);
        _mm256_storeu_ps(&result->data[i], vr);
    }

    for (; i < result->count; i++) {
        result->data[i] = t->data[i] - scalar;
    }

    return result;

}

Tensor* tensor_multiply_scalar(Tensor* t, float  scalar) {
    if (!t) return NULL;

    Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;

    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i < t->count - 8; i += 8) {
        __m256 vt = _mm256_loadu_ps(&t->data[i]);
        __m256 vr = _mm256_mul_ps(vt, vs);
        _mm256_storeu_ps(&result->data[i], vr);
    }

    for (; i < result->count; i++) {
        result->data[i] = t->data[i] * scalar;
    }

    return result;
}

void tensor_multiply_scalar_exsting(Tensor* dest, Tensor* source, float  scalar) {
    if (!dest || !source) return NULL;

    // Check if dimensions match
    if (dest->dims != source->dims) {
        fprintf(stderr, "Error: Tensor dimensions don't match for addition\n");
        return NULL;
    }

    // Check if shapes match
    for (int i = 0; i < source->dims; i++) {
        if (source->shape[i] != dest->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes don't match for addition\n");
            return NULL;
        }
    }

    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i <= source->count - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&source->data[i]);
        __m256 vd = _mm256_loadu_ps(&dest->data[i]);
        __m256 vr = _mm256_fmadd_ps(va, vs, vd);// (a * b) + c
        _mm256_storeu_ps(&dest->data[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < dest->count; i++) {
        dest->data[i] += source->data[i] * scalar;
    }
}

void tensor_multiply_scalar_existing_more(
    Tensor* dests[],      
    Tensor* sources[],    
    const float scalar[], 
    int amount)          
{
    for (int k = 0; k < amount; ++k)
    {
        if (!dests[k] || !sources[k]) {
            fprintf(stderr,
                "tensor_multiply_scalar_existing_more: NULL tensor at slot %d\n", k);
            return;
        }

        if (dests[k]->dims != sources[k]->dims) {
            fprintf(stderr,
                "tensor_multiply_scalar_existing_more: dim mismatch at slot %d\n", k);
            return;
        }

        for (int d = 0; d < dests[k]->dims; ++d) {
            if (dests[k]->shape[d] != sources[k]->shape[d]) {
                fprintf(stderr,
                    "tensor_multiply_scalar_existing_more: shape mismatch at slot %d\n", k);
                return;
            }
        }
    }

    for (int k = 0; k < amount; ++k)
    {
        __m256 vs = _mm256_set1_ps(scalar[k]);   

        int j = 0;
        int cnt = sources[k]->count;           

        for (; j <= cnt - 8; j += 8) {
            __m256 va = _mm256_loadu_ps(&sources[k]->data[j]); 
            __m256 vd = _mm256_loadu_ps(&dests[k]->data[j]);  
            __m256 vr = _mm256_fmadd_ps(va, vs, vd);         
            _mm256_storeu_ps(&dests[k]->data[j], vr);      
        }

        for (; j < cnt; ++j) {
            dests[k]->data[j] += sources[k]->data[j] * scalar[k];
        }
    }
}



Tensor* tensor_div_scalar(Tensor* t, float  scalar) {
    if (!t) return NULL;

    Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;

    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i < t->count - 8; i += 8) {
        __m256 vt = _mm256_loadu_ps(&t->data[i]);
        __m256 vr = _mm256_div_ps(vt, vs);
        _mm256_storeu_ps(&result->data[i], vr);
    }

    for (; i < result->count; i++) {
        result->data[i] = t->data[i] / scalar;
    }

    return result;
}

float  tensor_sum(Tensor* t) {
    if (!t) return 0.0;

    float  sum = 0.0;
    int i = 0;
    for (; i < t->count - 8; i+=8) {
        __m256 va = _mm256_loadu_ps(&t->data[i]);
        sum += sum8(va);
    }

    for (; i < t->count; i++) {
        sum += t->data[i];
    }

    return sum;
}

float  tensor_mean(Tensor* t) {
    if (!t || t->count == 0) return 0.0;

    return tensor_sum(t) / t->count;
}

void tensor_print(Tensor* t) {
    if (!t) {
        printf("NULL tensor\n");
        return;
    }

    printf("Tensor: dims=%d, shape=[", t->dims);
    for (int i = 0; i < t->dims; i++) {
        printf("%d", t->shape[i]);
        if (i < t->dims - 1) printf(", ");
    }
    printf("], count=%d\n", t->count);

    // Simple printing for 1D and 2D tensors
    if (t->dims == 1) {
        printf("[");
        for (int i = 0; i < t->shape[0]; i++) {
            printf("%.4f", t->data[i]);
            if (i < t->shape[0] - 1) printf(", ");
        }
        printf("]\n");
    }
    else if (t->dims == 2) {
        printf("[\n");
        for (int i = 0; i < t->shape[0]; i++) {
            printf("  [");
            for (int j = 0; j < t->shape[1]; j++) {
                int indices[2] = { i, j };
                printf("%.4f", tensor_get_element(t, indices));
                if (j < t->shape[1] - 1) printf(", ");
            }
            printf("]\n");
        }
        printf("]\n");
    }
    else {
        printf("(Tensor data omitted for dims > 2)\n");
    }
}
