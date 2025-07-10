#include "Tensor_Header.h"


void array_add(float *a, float *b, float * result, int count){
    //this code use a special cpu instruion that copy 8 elemnts at once(parallelism)
    int i = 0;
    for (; i <= count - 8; i+=8) { 
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < count; i++) {
        result[i] = a[i] + b[i];
    }
} 

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

    result->grad_func = function_create(2,(Tensor*[]) {a, b}, add_backward);
    result->is_leaf = false;
    if(a->requires_grad || b->requires_grad){
        result->requires_grad = true;
    }

    array_add(a->data,b->data,result->data,a->count);

    return result;
}

void add_backward(Function *f, Tensor *output_grad){
    for (int i = 0; i < f->num_inputs; i++){
        array_add(f->inputs[i]->grad, output_grad->grad, f->inputs[i]->grad, output_grad->count);
    }
}

void array_add_scalar(float *a, float  scalar, float *result, int count){
    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i < count - 8; i+=8) {
        __m256 vt = _mm256_loadu_ps(&a[i]);
        __m256 vr = _mm256_add_ps(vt, vs);
        _mm256_storeu_ps(&result[i], vr);
    }

    for (; i < count; i++) {
        result[i] = a[i] + scalar;
    }
}

Tensor* tensor_add_scalar(Tensor* t, float  scalar) {
    if (!t) return NULL;

    Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;

    Tensor* scalar_tensor = tensor_create(t->dims, t->shape);
    result->grad_func = function_create(2,(Tensor*[]) {t , scalar_tensor}, add_backward);
    if(t->requires_grad){
        result->requires_grad = true;
        result->is_leaf = false;
    }

    array_add_scalar(t->data, scalar, result->data, t->count);

    return result;
}

void tensor_add_scalar_inplace(Tensor* t, float  scalar) {
    if (!t) return;

    array_add_scalar(t->data, scalar, t->data, t->count);
}

void tensor_add_inplace(Tensor* target, Tensor* other) {
    if (!target || !other) return;

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

    array_add(target->data, other->data, target->data, target->count);
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
        array_add(target->data, others[i]->data, target->data, target->count);
    }
}

void array_subtract(float *a, float *b, float * result, int count){
    //this code use a special cpu instruion that copy 8 elemnts at once(parallelism)
    int i = 0;
    for (; i <= count - 8; i+=8) { 
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < count; i++) {
        result[i] = a[i] - b[i];
    }
}

void array_sub_scalar(float *a, float  scalar, float *result, int count){
    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i < count - 8; i+=8) {
        __m256 vt = _mm256_loadu_ps(&a[i]);
        __m256 vr = _mm256_sub_ps(vt, vs);
        _mm256_storeu_ps(&result[i], vr);
    }

    for (; i < count; i++) {
        result[i] = a[i] - scalar;
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

    result->grad_func = function_create(2,(Tensor*[]) {a, b}, subtract_backward);
    result->is_leaf = false;
    if(a->requires_grad || b->requires_grad){
        result->requires_grad = true;
    }

    array_subtract(a->data, b->data, result->data, a->count);

    return result;
}

void subtract_backward(Function* f, Tensor* output_grad){
    Tensor *sub_tensor = tensor_create(output_grad->dims,output_grad->shape);
    tensor_fill(sub_tensor, -1.0f);

    array_multiply(sub_tensor->data, output_grad->grad, sub_tensor->data, output_grad->count);

    for (int i = 0; i < f->num_inputs; i++){
        if(i % 2 == 1)
            array_add(f->inputs[i]->grad, sub_tensor->grad, f->inputs[i]->grad, sub_tensor->count);
        else
            array_add(f->inputs[i]->grad, output_grad->grad, f->inputs[i]->grad, output_grad->count);
    }
}

Tensor* tensor_subtract_scalar(Tensor* t, float  scalar) {
    if (!t) return NULL;

    Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;
    Tensor* scalar_tensor = tensor_create(t->dims, t->shape);
    result->grad_func = function_create(2,(Tensor*[]) {t , scalar_tensor}, subtract_backward);
    if(t->requires_grad){
        result->requires_grad = true;
        result->is_leaf = false;
    }

    array_sub_scalar(t->data, scalar, result->data, t->count);

    return result;

}

void tensor_subtract_inplace(Tensor* target, Tensor* other){
    if (!target || !other) return;

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

    array_subtract(target->data, other->data, target->data, target->count);
}


void array_multiply(float *a, float *b, float * result, int count){
    //this code use a special cpu instruion that copy 8 elemnts at once(parallelism)
    int i = 0;
    for (; i <= count - 8; i+=8) { 
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < count; i++) {
        result[i] = a[i] * b[i];
    }
}

void array_mul_scalar(float *a, float  scalar, float *result, int count){
    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i < count - 8; i+=8) {
        __m256 vt = _mm256_loadu_ps(&a[i]);
        __m256 vr = _mm256_mul_ps(vt, vs);
        _mm256_storeu_ps(&result[i], vr);
    }

    for (; i < count; i++) {
        result[i] = a[i] * scalar;
    }
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

    result->grad_func = function_create(2,(Tensor*[]) {a, b}, mul_backward);
    result->is_leaf = false;
    if(a->requires_grad || b->requires_grad){
        result->requires_grad = true;
    }

    array_multiply(a->data, b->data, result->data, a->count);

    return result;
}

void mul_backward(Function* f, Tensor* output_grad){
    Tensor *mul_sum = tensor_create(output_grad->dims, output_grad->shape);
    tensor_fill(mul_sum, 1.0f);
    
    for(int i=0; i < f->num_inputs ;i++){
        array_multiply(mul_sum->data, f->inputs[i]->data, mul_sum->data, output_grad->count);
    }

    float *mul_grad = (float*)malloc(output_grad->count * sizeof(float));
    for(int i=0; i < f->num_inputs; i++){
        fprintf(stderr,"\ni: %d\n", i);
        array_div(mul_sum->data, f->inputs[i]->data, mul_grad, output_grad->count);
        array_add(mul_grad, output_grad->grad, f->inputs[i]->grad, output_grad->count);
    }
    free(mul_grad);
    tensor_free(mul_sum);
}

void tensor_mul_scalar_inplace(Tensor* t, float scalar) {
    if (!t) return;

    array_mul_scalar(t->data, scalar, t->data, t->count);
}


Tensor* tensor_multiply_scalar(Tensor* t, float  scalar) {
    if (!t) return NULL;

    Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;
    Tensor* scalar_tensor = tensor_create(t->dims, t->shape);
    result->grad_func = function_create(2,(Tensor*[]) {t , scalar_tensor}, mul_backward);
    if(t->requires_grad){
        result->requires_grad = true;
        result->is_leaf = false;
    }

    array_mul_scalar(t->data, scalar, result->data, t->count);

    return result;
}

void tensor_multiply_scalar_exsting(Tensor* dest, Tensor* source, float  scalar) {
    if (!dest || !source) return;

    // Check if dimensions match
    if (dest->dims != source->dims) {
        fprintf(stderr, "Error: Tensor dimensions don't match for addition\n");
        return;
    }

    // Check if shapes match
    for (int i = 0; i < source->dims; i++) {
        if (source->shape[i] != dest->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes don't match for addition\n");
            return;
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

void array_div(float *a, float *b, float * result, int count){
    //this code use a special cpu instruion that copy 8 elemnts at once(parallelism)
    int i = 0;
    for (; i <= count - 8; i+=8) { 
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < count; i++) {
        result[i] = a[i] / b[i];
    }
}

void array_div_scalar(float *a, float  scalar, float *result, int count){
    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i < count - 8; i+=8) {
        __m256 vt = _mm256_loadu_ps(&a[i]);
        __m256 vr = _mm256_div_ps(vt, vs);
        _mm256_storeu_ps(&result[i], vr);
    }

    for (; i < count; i++) {
        result[i] = a[i] / scalar;
    }
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    if (a->dims != b->dims) {
        fprintf(stderr, "Error: Tensor dimensions don't match for tensor_div\n");
        return NULL;
    }

    for (int i = 0; i < a->dims; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes don't match for tensor_div\n");
            return NULL;
        }
    }

    // Create result tensor
    Tensor* result = tensor_create(a->dims, a->shape);
    if (!result) return NULL;
    result->grad_func = function_create(2,(Tensor*[]) {a, b}, div_backward);
    result->is_leaf = false;
if(a->requires_grad || b->requires_grad){
        result->requires_grad = true;
    }

    array_div(a->data, b->data, result->data, a->count);

    return result;
}

Tensor* tensor_div_scalar(Tensor* t, float  scalar) {
    if (!t) return NULL;

    Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;
    
    Tensor* scalar_tensor = tensor_create(t->dims, t->shape);
    result->grad_func = function_create(2,(Tensor*[]) {t , scalar_tensor}, div_backward);
    if(t->requires_grad){
        result->requires_grad = true;
        result->is_leaf = false;
    }

    array_div_scalar(t->data, scalar, result->data, t->count);
    return result;
}

void div_backward(Function* f, Tensor* output_grad){
    Tensor *div_sum = tensor_create(output_grad->dims, output_grad->shape);
    tensor_fill(div_sum, 1.0f);
    for(int i=0; i<f->num_inputs;i++){
        array_div(div_sum->data, f->inputs[i]->data, div_sum->data, output_grad->count);
    }

    float *div_grad = (float*)malloc(output_grad->count * sizeof(float));
    for(int i=0; i<f->num_inputs;i++){
        array_multiply(div_sum->data, f->inputs[i]->data, div_grad, output_grad->count);
        array_add(div_grad, output_grad->grad, f->inputs[i]->grad, output_grad->count);
    }
    tensor_free(div_sum);
    free(div_grad);
}

void tensor_div_scalar_inplace(Tensor* t, float  scalar) {
    if (!t) return;

    int i = 0;
    __m256 vs = _mm256_set1_ps(scalar);
    for (; i <= t->count - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&t->data[i]);
        __m256 vr = _mm256_div_ps(va, vs);
        _mm256_storeu_ps(&t->data[i], vr);
    }

    // must add it because the loop before stops 7 elemnts or less before the end
    for (; i < t->count; i++) {
        t->data[i] /= scalar;
    } 
}

float sum8(__m256 v) {
    __m256 hsum = _mm256_hadd_ps(v, v);
    hsum = _mm256_hadd_ps(hsum, hsum);
    __m128 sum128 = _mm256_extractf128_ps(hsum, 1);
    sum128 = _mm_add_ps(sum128, _mm256_castps256_ps128(hsum));
    return _mm_cvtss_f32(sum128);
}

void matmul(float* a, float* b, float* result, int a_dim, int b_dim, int same_dim) {
    // assuming b is transposed for faster calc
    for (int i = 0; i < a_dim; i++) {
        for (int j = 0; j < b_dim; j++) {
            float sum = 0.0;
            int k = 0;
            
            // Vectorized loop - process 8 elements at a time
            for (; k <= same_dim - 8; k += 8) {
                __m256 va = _mm256_loadu_ps(&a[i * same_dim + k]);
                __m256 vb = _mm256_loadu_ps(&b[j * same_dim + k]);
                __m256 vr = _mm256_mul_ps(va, vb);
                sum += sum8(vr);
            }

            // Handle remaining elements
            for (; k < same_dim; k++) {
                sum += a[i * same_dim + k] * b[j * same_dim + k];
            }
            result[i * b_dim + j] = sum;
        }
    }
}

Tensor* tensor_mmul(Tensor* a, Tensor* b, bool transposed) {
    if (!a || !b) return NULL;

    int adims = a->dims;
    int bdims = b->dims;
    int rdims = 2;
    int result_shape[3];

    // Normalize 1D to 2D
    int a_rows, a_cols, b_rows, b_cols, r_rows, r_cols;
    int a_batch = 1, b_batch = 1;

    if (adims == 1) {
        a_rows = 1;
        a_cols = a->shape[0];
    } else {
        a_rows = a->shape[adims - 2];
        a_cols = a->shape[adims - 1];
    }

    if (bdims == 1) {
        b_rows = b->shape[0];
        b_cols = 1;
    } else {
        b_rows = b->shape[bdims - 2];
        b_cols = b->shape[bdims - 1];
    }

    // Determine result dimensions first
    if (transposed) {
        r_rows = a_rows;
        r_cols = b_rows;
    } else {
        r_rows = a_rows;
        r_cols = b_cols;
    }

    // Batch size for 3D tensors
    if (adims == 3 || bdims == 3) {
        if (adims == 3 && bdims == 3) {
            a_batch = a->shape[0];
            b_batch = b->shape[0];

            if (a_batch != b_batch) {
                fprintf(stderr, "Error: Mismatched batch dimensions\n");
                return NULL;
            }
        }else if (adims == 3 && bdims != 3){
            a_batch = a->shape[0];
        }else{
            b_batch = b->shape[0];
        }
        
        rdims = 3;
        result_shape[0] = (a_batch > b_batch) ? a_batch : b_batch;
        result_shape[1] = r_rows;
        result_shape[2] = r_cols;
    } else {
        result_shape[0] = r_rows;
        result_shape[1] = r_cols;
    }

    // Validate dimensions
    if (!transposed) {
        if (a_cols != b_rows) {
            fprintf(stderr, "Error: Cannot multiply matrices with shapes [%d, %d] and [%d, %d]\n", 
                    a_rows, a_cols, b_rows, b_cols);
            return NULL;
        }
    } else {
        if (a_cols != b_cols) {
            fprintf(stderr, "Error: Cannot multiply matrices with shapes [%d, %d] and [%d, %d]T\n", 
                    a_rows, a_cols, b_rows, b_cols);
            return NULL;
        }
    }

    Tensor* result = tensor_create(rdims, result_shape);
    if (!result) return NULL;
        
    result->grad_func = function_create(2, (Tensor*[]) {a, b}, matmul_backward);
    result->is_leaf = false;
    if (a->requires_grad || b->requires_grad) {
        result->requires_grad = true;
    }

    // Perform multiplication
    if (!transposed) {
        Tensor* b_t = tensor_transpose(b);
        if (!b_t) {
            tensor_free(result);
            return NULL;
        }
        if (a_batch == 1 && b_batch == 1) {
            matmul(a->data, b_t->data, result->data, a_rows, b_cols, a_cols);
        } else {
            int result_batch = (a_batch > b_batch) ? a_batch : b_batch;
        
            for (int k = 0; k < result_batch; k++) {
                int a_idx = (a_batch == 1) ? 0 : k;
                int b_idx = (b_batch == 1) ? 0 : k;
                
                float* a_slice = (a_batch == 1) ? a->data : &a->data[a_idx * a->strides[0]];
                float* b_slice = (b_batch == 1) ? b_t->data : &b_t->data[b_idx * b_t->strides[0]];
                float* result_slice = &result->data[k * result->strides[0]];
                
                matmul(a_slice, b_slice, result_slice, a_rows, b_cols, a_cols);
            }
        }
        tensor_free(b_t);
    } else {
        if (a_batch == 1 && b_batch == 1) {
            matmul(a->data, b->data, result->data, a_rows, b_rows, a_cols);
        } else {
            int result_batch = (a_batch > b_batch) ? a_batch : b_batch;
        
            for (int k = 0; k < result_batch; k++) {
                int a_idx = (a_batch == 1) ? 0 : k;
                int b_idx = (b_batch == 1) ? 0 : k;
                
                float* a_slice = (a_batch == 1) ? a->data : &a->data[a_idx * a->strides[0]];
                float* b_slice = (b_batch == 1) ? b->data : &b->data[b_idx * b->strides[0]];
                float* result_slice = &result->data[k * result->strides[0]];
                
                matmul(a_slice, b_slice, result_slice, a_rows, b_rows, a_cols);
            }
        }
    }

    return result;
}

void matmul_backward(Function* f, Tensor* output_grad) {
    // For matrix multiplication C = A @ B:
    // grad_A = grad_C @ B.T
    // grad_B = A.T @ grad_C
    
    Tensor* a = f->inputs[0];
    Tensor* b = f->inputs[1];
    
    // Create a copy of output gradient for computation
    Tensor* grad = tensor_create(output_grad->dims, output_grad->shape);
    memcpy(grad->data, output_grad->grad, sizeof(float) * output_grad->count);
    
    if (a->requires_grad) {
        Tensor* b_for_grad = b;
        bool should_free_b = false;
        
        // Handle 1D case by reshaping
        if (b->dims == 1) {
            b_for_grad = tensor_reshape(b, 2, (int[]) {1, b->shape[0]});
            should_free_b = true;
        }
        
        // grad_A = grad_C @ B.T
        //Tensor* b_t = tensor_transpose(b_for_grad);
        Tensor* grad_a = tensor_mmul(grad, b_for_grad, true);
        //tensor_free(b_t);

        // Handle broadcasting: if original A had fewer batch dimensions, sum across batches
        if (a->dims < grad_a->dims) {
            // Sum across the batch dimension to match original A shape
            Tensor* grad_a_summed = tensor_sum_axis(grad_a, 0);
            tensor_free(grad_a);
            grad_a = grad_a_summed;
        }
        // If A was broadcasted (batch size 1 but result has larger batch), sum the gradient
        else if (a->dims == 3 && grad_a->dims == 3 && a->shape[0] == 1 && grad_a->shape[0] > 1) {
            Tensor* grad_a_summed = tensor_sum_axis(grad_a, 0);
            tensor_free(grad_a);
            grad_a = grad_a_summed;
        }
        
        if (a->grad) {
            // Accumulate gradients
            array_add(a->grad, grad_a->data, a->grad, grad_a->count);
            tensor_free(grad_a);
        } else {
            // Initialize gradient
            a->grad = malloc(sizeof(float) * a->count);
            memcpy(a->grad, grad_a->data, sizeof(float) * a->count);
            tensor_free(grad_a);
        }
        
        if (should_free_b) {
            tensor_free(b_for_grad);
        }
    }
    
    if (b->requires_grad) {
        Tensor* a_for_grad = a;
        bool should_free_a = false;
        
        // Handle 1D case by reshaping
        if (a->dims == 1) {
            a_for_grad = tensor_reshape(a, 2, (int[]) {1, a->shape[0]});
            should_free_a = true;
        }
        
        // grad_B = A.T @ grad_C
        Tensor* a_t = tensor_transpose(a_for_grad);
        Tensor* grad_b = tensor_mmul(a_t, grad, false);
        tensor_free(a_t);
        
        // Handle broadcasting: if original B had fewer batch dimensions, sum across batches
        if (b->dims < grad_b->dims) {
            // Sum across the batch dimension to match original B shape
            Tensor* grad_b_summed = tensor_sum_axis(grad_b, 0);
            tensor_free(grad_b);
            grad_b = grad_b_summed;
        }
        // If B was broadcasted (batch size 1 but result has larger batch), sum the gradient
        else if (b->dims == 3 && grad_b->dims == 3 && b->shape[0] == 1 && grad_b->shape[0] > 1) {
            Tensor* grad_b_summed = tensor_sum_axis(grad_b, 0);
            tensor_free(grad_b);
            grad_b = grad_b_summed;
        }
        
        if (b->grad) {
            // Accumulate gradients
            array_add(b->grad, grad_b->data, b->grad, grad_b->count);
            tensor_free(grad_b);
        } else {
            // Initialize gradient
            b->grad = malloc(sizeof(float) * b->count);
            memcpy(b->grad, grad_b->data, sizeof(float) * b->count);
            tensor_free(grad_b);
        }
        
        if (should_free_a) {
            tensor_free(a_for_grad);
        }
    }
    
    tensor_free(grad);
}

void tensor_brodcast_inplace(Tensor* target, Tensor* other){
    if (!target || !other) return;

    if (target->dims > other->dims + 1 && target->shape[0] == other->shape[0]){//because other is 1d array
        for (int i = 0; i < target->shape[0]; i++){
            int j = 0;
            __m256 vb = _mm256_set1_ps(other->data[j]);
            for (; j <= target->shape[1] - 8; j += 8) {
                __m256 va = _mm256_loadu_ps(&target->data[i * target->shape[0] + j]);
                __m256 vr = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(&target->data[j], vr);
            }

            for (; j < target->count; j++) {
                target->data[j] += other->data[j];
            }
        }
        
    }
    
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

float  tensor_sum(Tensor* t) {
    if (!t) return 0.0;

    float sum = 0.0;
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
