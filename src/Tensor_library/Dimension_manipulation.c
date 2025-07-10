#include "Tensor_Header.h"

void transpose_blocked(const float* src, float* dst, int rows, int cols) {
    const int BLOCK_SIZE = 8;
    
    for (int i = 0; i < rows; i += BLOCK_SIZE) {
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            int max_i = (i + BLOCK_SIZE > rows) ? rows : i + BLOCK_SIZE;
            int max_j = (j + BLOCK_SIZE > cols) ? cols : j + BLOCK_SIZE;

            // Try to process 8x8 blocks with AVX2 if available
            if (max_i - i >= 8 && max_j - j >= 8) {
                transpose_8x8_avx2(&src[i * cols + j], &dst[j * rows + i], cols, rows);
            }
            // Try 4x4 blocks with SSE
            else {
                for (int bi = i; bi < max_i; bi += 4) {
                    for (int bj = j; bj < max_j; bj += 4) {
                        if (bi + 4 <= max_i && bj + 4 <= max_j) {
                            transpose_4x4_sse(&src[bi * cols + bj], &dst[bj * rows + bi], cols, rows);
                        } else {
                            // Scalar fallback for edge cases
                            for (int ii = bi; ii < max_i; ++ii) {
                                for (int jj = bj; jj < max_j; ++jj) {
                                    dst[jj * rows + ii] = src[ii * cols + jj];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Optimized 4x4 transpose using SSE
static inline void transpose_4x4_sse(const float* src, float* dst, int src_stride, int dst_stride) {
    __m128 row0 = _mm_loadu_ps(&src[0 * src_stride]);
    __m128 row1 = _mm_loadu_ps(&src[1 * src_stride]);
    __m128 row2 = _mm_loadu_ps(&src[2 * src_stride]);
    __m128 row3 = _mm_loadu_ps(&src[3 * src_stride]);

    __m128 tmp0 = _mm_unpacklo_ps(row0, row1);
    __m128 tmp1 = _mm_unpacklo_ps(row2, row3);
    __m128 tmp2 = _mm_unpackhi_ps(row0, row1);
    __m128 tmp3 = _mm_unpackhi_ps(row2, row3);

    row0 = _mm_movelh_ps(tmp0, tmp1);
    row1 = _mm_movehl_ps(tmp1, tmp0);
    row2 = _mm_movelh_ps(tmp2, tmp3);
    row3 = _mm_movehl_ps(tmp3, tmp2);

    _mm_storeu_ps(&dst[0 * dst_stride], row0);
    _mm_storeu_ps(&dst[1 * dst_stride], row1);
    _mm_storeu_ps(&dst[2 * dst_stride], row2);
    _mm_storeu_ps(&dst[3 * dst_stride], row3);
}

// Optimized 8x8 transpose using AVX2
static inline void transpose_8x8_avx2(const float* src, float* dst, int src_stride, int dst_stride) {
    __m256 row0 = _mm256_loadu_ps(&src[0 * src_stride]);
    __m256 row1 = _mm256_loadu_ps(&src[1 * src_stride]);
    __m256 row2 = _mm256_loadu_ps(&src[2 * src_stride]);
    __m256 row3 = _mm256_loadu_ps(&src[3 * src_stride]);
    __m256 row4 = _mm256_loadu_ps(&src[4 * src_stride]);
    __m256 row5 = _mm256_loadu_ps(&src[5 * src_stride]);
    __m256 row6 = _mm256_loadu_ps(&src[6 * src_stride]);
    __m256 row7 = _mm256_loadu_ps(&src[7 * src_stride]);

    // Step 1: Transpose 4x4 blocks
    __m256 tmp0 = _mm256_unpacklo_ps(row0, row1);
    __m256 tmp1 = _mm256_unpackhi_ps(row0, row1);
    __m256 tmp2 = _mm256_unpacklo_ps(row2, row3);
    __m256 tmp3 = _mm256_unpackhi_ps(row2, row3);
    __m256 tmp4 = _mm256_unpacklo_ps(row4, row5);
    __m256 tmp5 = _mm256_unpackhi_ps(row4, row5);
    __m256 tmp6 = _mm256_unpacklo_ps(row6, row7);
    __m256 tmp7 = _mm256_unpackhi_ps(row6, row7);

    // Step 2: Transpose 2x2 blocks
    __m256 tmp8 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tmp9 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 tmp10 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tmp11 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 tmp12 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tmp13 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 tmp14 = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tmp15 = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(3, 2, 3, 2));

    // Step 3: Final transpose with permute
    row0 = _mm256_permute2f128_ps(tmp8, tmp12, 0x20);
    row1 = _mm256_permute2f128_ps(tmp9, tmp13, 0x20);
    row2 = _mm256_permute2f128_ps(tmp10, tmp14, 0x20);
    row3 = _mm256_permute2f128_ps(tmp11, tmp15, 0x20);
    row4 = _mm256_permute2f128_ps(tmp8, tmp12, 0x31);
    row5 = _mm256_permute2f128_ps(tmp9, tmp13, 0x31);
    row6 = _mm256_permute2f128_ps(tmp10, tmp14, 0x31);
    row7 = _mm256_permute2f128_ps(tmp11, tmp15, 0x31);

    _mm256_storeu_ps(&dst[0 * dst_stride], row0);
    _mm256_storeu_ps(&dst[1 * dst_stride], row1);
    _mm256_storeu_ps(&dst[2 * dst_stride], row2);
    _mm256_storeu_ps(&dst[3 * dst_stride], row3);
    _mm256_storeu_ps(&dst[4 * dst_stride], row4);
    _mm256_storeu_ps(&dst[5 * dst_stride], row5);
    _mm256_storeu_ps(&dst[6 * dst_stride], row6);
    _mm256_storeu_ps(&dst[7 * dst_stride], row7);
}

Tensor* tensor_transpose(Tensor* t) {
    int* result_shape = malloc(sizeof(int) * t->dims);
    if (!result_shape) return NULL;
    memcpy(result_shape, t->shape, sizeof(int) * t->dims);
    
    // Safe swap of last two dimensions
    int temp = result_shape[t->dims - 1];
    result_shape[t->dims - 1] = result_shape[t->dims - 2];
    result_shape[t->dims - 2] = temp;

    Tensor* result = tensor_create(t->dims, result_shape);
    if (!result) {
        free(result_shape);
        return NULL;
    }
    free(result_shape);
    
    if(t->dims == 2) {
        transpose_blocked(t->data, result->data, t->shape[0], t->shape[1]);
    } else if (t->dims == 3) {
        // Fixed: properly iterate through each slice
        for(int i = 0; i < t->shape[0]; i++) {
            transpose_blocked(&t->data[i * t->strides[0]], 
                            &result->data[i * result->strides[0]], 
                            t->shape[1], t->shape[2]);
        }
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

    int temp = t->shape[t->dims - 2];
    t->shape[t->dims - 2] = t->shape[t->dims - 1];
    t->shape[t->dims - 1] = temp;

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

Tensor* tensor_sum_axis(Tensor* t, int axis) {
    if (axis < 0 || axis >= t->dims) return NULL;
    
    // Create result shape by removing the specified axis
    int* result_shape = malloc(sizeof(int) * (t->dims - 1));
    int result_dims = t->dims - 1;
    
    for (int i = 0, j = 0; i < t->dims; i++) {
        if (i != axis) {
            result_shape[j++] = t->shape[i];
        }
    }
    
    Tensor* result = tensor_create(result_dims, result_shape);
    free(result_shape);
    
    if (!result) return NULL;
    
    // Initialize result to zero
    memset(result->data, 0, sizeof(float) * result->count);
    
    // Sum along the specified axis
    if (axis == 0 && t->dims == 3) {
        // Sum across batch dimension
        for (int b = 0; b < t->shape[0]; b++) {
            for (int i = 0; i < t->shape[1]; i++) {
                for (int j = 0; j < t->shape[2]; j++) {
                    int src_idx = b * t->strides[0] + i * t->strides[1] + j * t->strides[2];
                    int dst_idx = i * result->strides[0] + j * result->strides[1];
                    result->data[dst_idx] += t->data[src_idx];
                }
            }
        }
    }
    // Add other axis cases as needed
    
    return result;
}