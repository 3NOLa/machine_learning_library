#include "active_functions.h"
#include "Tensor_library/Tensor_Header.h"
#include <math.h>

Tensor* RELu_function(Tensor* t){
	Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;
        
    result->grad_func = function_create(1, (Tensor*[]) {t}, RELu_derivative_function);
    result->is_leaf = false;
    if (t->requires_grad) {
        result->requires_grad = true;
    }

	int i = 0;
	__m256 zero = _mm256_setzero_ps();
	for(; i < t->count; i+=8){
		__m256 va = _mm256_loadu_ps(&t->data[i]);
		__m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OQ);
		__m256 vr = _mm256_and_ps(va, mask);
		_mm256_storeu_ps(&result[i], vr);
	}

	for(; i < t->count; i++){
		result->data[i] = (t->data[i] > 0) ? t->data[i] : 0;
	}

	return result;
}


void RELu_derivative_function(Function* f, Tensor* output_grad){
	int i = 0;
	__m256 zero = _mm256_setzero_ps();
	for(; i < output_grad->count; i+=8){
		__m256 va = _mm256_loadu_ps(&output_grad->data[i]);
		__m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OQ);
		_mm256_storeu_ps(&f->inputs[0]->grad[i], mask);
	}

	for(; i < f->inputs[0]->count; i++){
		f->inputs[0]->grad[i] = (f->inputs[0]->grad[i] > 0) ? 1 : 0;
	}
}


Tensor* leaky_RELu_function(Tensor* t){
	Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;
        
    result->grad_func = function_create(1, (Tensor*[]) {t}, RELu_derivative_function);
    result->is_leaf = false;
    if (t->requires_grad) {
        result->requires_grad = true;
    }

	int i = 0;
	__m256 zero = _mm256_setzero_ps();
	__m256 epsilon = _mm256_set1_ps((float)1e-15);
	for(; i < t->count; i+=8){
		__m256 va = _mm256_loadu_ps(&t->data[i]);
		__m256 epvr = _mm256_mul_ps(epsilon, va);

		__m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OQ);
		__m256 vr = _mm256_blendv_ps(va, epvr, mask);
		_mm256_storeu_ps(&result[i], vr);
	}

	for(; i < t->count; i++){
		result->data[i] = (t->data[i] > 0) ? t->data[i] :  t->data[i] * (float)1e-15;
	}

	return result;
}

void leaky_RELu_derivative_function(Function* f, Tensor* output_grad){
	int i = 0;
	__m256 onev = _mm256_set1_ps(1.0f);
	__m256 epsilon = _mm256_set1_ps((float)1e-15);
	__m256 zero = _mm256_setzero_ps();
	for(; i < output_grad->count; i+=8){
		__m256 va = _mm256_loadu_ps(&f->inputs[0]->data[i]);
		__m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OQ);
		__m256 vr = _mm256_blendv_ps(onev, epsilon, mask);
		_mm256_storeu_ps(&f->inputs[0]->grad[i], vr);
	}

	for(; i < f->inputs[0]->count; i++){
		f->inputs[0]->grad[i] = (f->inputs[0]->data[i] > 0) ? 1 : (float)1e-15;
	}
}


Tensor* Sigmoid_function(Tensor* t){
	Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;
        
    result->grad_func = function_create(1, (Tensor*[]) {t}, Sigmoid_derivative_function);
    result->is_leaf = false;
    if (t->requires_grad) {
        result->requires_grad = true;
    }

	int i = 0;
	__m256 onev = _mm256_set1_ps(1.0f);
	for(; i < t->count; i+=8){
		__m256 va = _mm256_loadu_ps(&t->data[i]);
		__m256 mva = _mm256_mul_ps(va,  _mm256_set1_ps(-1.0f)); // to get -value
		__m256 vexp = _mm256_exp_ps(mva); // to get e^(-x)
		__m256 pvexp = _mm256_add_ps(vexp, onev); //(1 + e^-(x))
		__m256 vr = _mm256_div_ps(onev, pvexp); //(1 + e^-(x))
		_mm256_storeu_ps(&result[i], vr);
	}

	for(; i < t->count; i++){
		result->data[i] = 1 / (1 + exp(-t->data[i]));;
	}
	return result;
}

void Sigmoid_derivative_function(Function* f, Tensor* output_grad){
	int i = 0;
	__m256 onev = _mm256_set1_ps(1.0f);
	for(; i < output_grad->count; i+=8){
		__m256 va = _mm256_loadu_ps(&output_grad->data[i]);
		__m256 vam = _mm256_sub_ps(onev, va);
		__m256 vr = _mm256_mul_ps(va, vam);
		_mm256_storeu_ps(&f->inputs[0]->grad[i], vr);
	}

	for(; i < f->inputs[0]->count; i++){
		f->inputs[0]->grad[i] = output_grad->data[i] * (1 - output_grad->data[i]);
	}
}

Tensor* Tanh_function(Tensor* t){
	Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;
        
    result->grad_func = function_create(1, (Tensor*[]) {t}, Tanh_derivative_function);
    result->is_leaf = false;
    if (t->requires_grad) {
        result->requires_grad = true;
    }

	int i = 0;
	for(; i < t->count; i+=8){
		__m256 va = _mm256_loadu_ps(&t->data[i]);
		__m256 vexp = _mm256_exp_ps(va); // to get e^(x) = x

		__m256 mva = _mm256_mul_ps(va,  _mm256_set1_ps(-1.0f)); // to get -value
		__m256 vexpm = _mm256_exp_ps(mva); // to get e^(-x) = y

		__m256 vplus = _mm256_add_ps(vexp, vexpm); // x + y
		__m256 vminus = _mm256_sub_ps(vexp, vexpm); // x - y

		__m256 vr = _mm256_div_ps(vminus, vplus); // (x - y) / (x + y)
		_mm256_storeu_ps(&result[i], vr);
	}

	for(; i < t->count; i++){
		float  x = exp(t->data[i]);
		float  y = exp(-t->data[i]);  
		result->data[i] = (x - y) / (x + y);
	}
	return result;
}

void Tanh_derivative_function(Function* f, Tensor* output_grad){
	int i = 0;
	__m256 onev = _mm256_set1_ps(1.0f);
	for(; i < output_grad->count; i+=8){
		__m256 va = _mm256_loadu_ps(&output_grad->data[i]);
		__m256 v2 = _mm256_mul_ps(va, va);
		__m256 vr = _mm256_sub_ps(onev, va);
		_mm256_storeu_ps(&f->inputs[0]->grad[i], vr);
	}

	for(; i < f->inputs[0]->count; i++){
		f->inputs[0]->grad[i] =  1 - output_grad->data[i] * output_grad->data[i];
	}
}

Tensor* linear_function(Tensor* t){
	Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;
        
    result->grad_func = function_create(1, (Tensor*[]) {t}, Tanh_derivative_function);
    result->is_leaf = false;
    if (t->requires_grad) {
        result->requires_grad = true;
    }
	tensor_copy(result, t);
	return result;
}

void linear_derivative_function(Function* f, Tensor* output_grad){
	int i = 0;
	__m256 onev = _mm256_set1_ps(1.0f);
	for(; i < output_grad->count; i+=8){
		_mm256_storeu_ps(&f->inputs[0]->grad[i], onev);
	}

	for(; i < f->inputs[0]->count; i++){
		f->inputs[0]->grad[i] =  1;
	}
}

Tensor* gelu_function(Tensor* t){
	Tensor* result = tensor_create(t->dims, t->shape);
    if (!result) return NULL;
        
    result->grad_func = function_create(1, (Tensor*[]) {t}, gelu_derivative_function);
    result->is_leaf = false;
    if (t->requires_grad) {
        result->requires_grad = true;
    }

	int i = 0;
	__m256 half_vec = _mm256_set1_ps(0.5f);
	__m256 one_vec = _mm256_set1_ps(1.0f);
	__m256 pi_vec = _mm256_set1_ps(3.14159265358979323846f);
	__m256 div_pi_2 = _mm256_mul_ps(_mm256_set1_ps(2.0f), pi_vec); //(2 / PI)
	__m256 sqrt_pi = _mm256_sqrt_ps(div_pi_2);//sqrt(2 / PI)

	__m256 var = _mm256_set1_ps(0.044715);
	for(; i < t->count; i+=8){
		__m256 va = _mm256_loadu_ps(&t->data[i]);
		__m256 va_3 = _mm256_mul_ps(var, _mm256_mul_ps(va, _mm256_mul_ps(va, va))); // to get 0.044715 * value * value * value
		__m256 va_va_3 = _mm256_add_ps(va, va_3); // to get value + 0.044715 * value * value * value

		__m256 vr = _mm256_mul_ps(sqrt_pi, va_va_3); // to get sqrt(2 / PI) * (value + 0.044715 * value * value * value);

		//doing tanh to the result for better preformnce instead of calling tanh function
		__m256 vexp = _mm256_exp_ps(vr); // to get e^(x) = x

		__m256 mva = _mm256_mul_ps(vr,  _mm256_set1_ps(-1.0f)); // to get -value
		__m256 vexpm = _mm256_exp_ps(mva); // to get e^(-x) = y

		__m256 vplus = _mm256_add_ps(vexp, vexpm); // x + y
		__m256 vminus = _mm256_sub_ps(vexp, vexpm); // x - y

		__m256 vrtan = _mm256_div_ps(vminus, vplus); // (x - y) / (x + y)
		// ended tanh

		__m256 vresult = _mm256_mul_ps(_mm256_mul_ps(half_vec, va), _mm256_add_ps(vrtan, one_vec)); // 0.5 * value * (1 + Tanh_function(s));
		_mm256_storeu_ps(&result[i], vresult);
	}

	for(; i < t->count; i++){
		float mid = sqrt(2 / 3.14159265358979323846f) * (t->data[i] + 0.044715 * t->data[i] * t->data[i] * t->data[i]);

		float  x = exp(mid);
		float  y = exp(-mid);  
		float rtanh = (x - y) / (x + y);

		result->data[i] = 0.5 * t->data[i] * (1 + rtanh);
	}

	return result;
}

void gelu_derivative_function(Function* f, Tensor* output_grad){
	int i = 0;
	__m256 half_vec = _mm256_set1_ps(0.5f);
	__m256 one_vec = _mm256_set1_ps(1.0f);
	__m256 two_vec = _mm256_set1_ps(2.0f);
	__m256 three_vec = _mm256_set1_ps(3.0f);
	__m256 pi_vec = _mm256_set1_ps(3.14159265358979323846f);
	__m256 div_pi_2 = _mm256_mul_ps(two_vec, pi_vec); //(2 / PI)
	__m256 sqrt_pi = _mm256_sqrt_ps(div_pi_2);

	__m256 var = _mm256_set1_ps(0.044715);
	for(; i < output_grad->count; i+=8){
		__m256 va = _mm256_loadu_ps(&output_grad->data[i]);
		__m256 vi = _mm256_loadu_ps(&f->inputs[0]->data[i]);

		//tanh = (2 * output) / input - 1;
		__m256 vi2 = _mm256_mul_ps(va, two_vec);
		__m256 vtanh = _mm256_div_ps(vi2, _mm256_sub_ps(vi, one_vec));

		//(1 + 3 * 0.044715 * input * input)
		__m256 v3in = _mm256_mul_ps(_mm256_mul_ps(three_vec, var), _mm256_mul_ps(vi, vi));
		__m256 v3pusone = _mm256_add_ps(v3in, one_vec);

		//(1 - tanh * tanh)
		__m256 one_tanh2 = _mm256_sub_ps(one_vec, _mm256_mul_ps(vtanh, vtanh));

		//0.5 * input
		__m256 half_input = _mm256_sub_ps(half_vec, vi);

		//0.5 * input * (1 - tanh * tanh) * sqrt(2 / M_PI) * (1 + 3 * 0.044715 * input * input);
		__m256 vmost = _mm256_mul_ps(_mm256_mul_ps(half_input, one_tanh2), _mm256_mul_ps(sqrt_pi, v3pusone));

		
		//0.5 * (1 + tanh)
		__m256 half_one_tanh = _mm256_mul_ps(half_vec, _mm256_add_ps(one_vec, vtanh));

		// 0.5 * (1 + tanh) + vmost
		__m256 vr = _mm256_add_ps(half_one_tanh, vmost);

		_mm256_storeu_ps(&f->inputs[0]->grad[i], vr);
	}

	for(; i < f->inputs[0]->count; i++){
		float tanh = (2 * output_grad->data[i]) / f->inputs[0]->data[i] - 1;
		f->inputs[0]->grad[i] =  0.5 * (1 + tanh) + 0.5 * f->inputs[0]->data[i] * (1 - tanh * tanh) * sqrt(2 / 3.14159265358979323846f) * (1 + 3 * 0.044715 * f->inputs[0]->data[i] * f->inputs[0]->data[i]);
	}
}

Tensor* swish_function(Tensor* t){
	Tensor *sigmoid = Sigmoid_function(t);
	sigmoid->grad_func->backward = swish_derivative_function;

	array_multiply(sigmoid->data, t->data, sigmoid->data, t->count);

	return sigmoid;
}

void swish_derivative_function(Function* f, Tensor* output_grad){
	Sigmoid_derivative_function(f, output_grad);
	array_multiply(f->inputs[0]->grad, f->inputs[0]->data, f->inputs[0]->grad, f->inputs[0]->count); //Sigmoid_function(input) + Sigmoid_derivative_function(n) * input;
	
	Tensor *sigmoid = Sigmoid_function(f->inputs[0]);//Sigmoid_function(input) +
	array_add(f->inputs[0]->grad, sigmoid->data, f->inputs[0]->grad, f->inputs[0]->count);
}