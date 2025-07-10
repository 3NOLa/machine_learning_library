#include "active_functions.h"
#include "neuron.h"
#include <math.h>

float  RELu_function(float  value)
{
	return (value > 0) ? value : 0;
}

float  leaky_RELu_function(float  value)
{
	return (value > 0) ? value : 0.01 * value;
}

float  Sigmoid_function(float  value)
{
	return 1 / (1 + exp(-value)); //s(x) = 1 / (1 + e^-(x))
}

float  Tanh_function(float  value)
{
	float  x = exp(value);
	float  y = exp(-value);
	return (x - y) / (x + y);
}

float  linear_function(float  value)
{
	return value;
}

float  gelu_function(float  value)
{
	float  s = sqrt(2 / M_PI) * (value + 0.044715 * value * value * value);
	return 0.5 * value * (1 + Tanh_function(s));
}

float  swish_function(float  value)
{
	return value * Sigmoid_function(value);
}

float  (*ActivationTypeMap(ActivationType function))(float )
{
	static void (*map[])(float ) = { 
		RELu_function, 
		leaky_RELu_function, 
		Sigmoid_function, 
		Tanh_function,
		linear_function,
		gelu_function,
		swish_function };
	return map[function];
}

float  RELu_derivative_function(neuron* n)
{
	return (n->output > 0) ? 1 : 0;
}

float  leaky_RELu_derivative_function(neuron* n)
{
	return (n->output > 0) ? 1 : 0.01;
}

float  Sigmoid_derivative_function(neuron* n)
{
	return n->output * (1 - n->output); 
}

float  Tanh_derivative_function(neuron* n)
{
	return 1 - n->output * n->output; // 1 - tanh^2
}

float  linear_derivative_function(neuron* n)
{
	return 1;
}
	
float  gelu_derivative_function(neuron* n)
{
	float  tanh = (2 * n->output) / n->pre_activation - 1;
	return 0.5 * (1 + tanh) + 0.5 * n->pre_activation * (1 - tanh * tanh) * sqrt(2 / M_PI) * (1 + 3 * 0.044715 * n->pre_activation * n->pre_activation);
}

float  swish_derivative_function(neuron* n)
{
	return Sigmoid_function(n->pre_activation) + Sigmoid_derivative_function(n) * n->pre_activation;
}

float  (*ActivationTypeDerivativeMap(ActivationType function))(neuron*)
{
	static float  (*map[])(neuron*) = {
		RELu_derivative_function,
		leaky_RELu_derivative_function,
		Sigmoid_derivative_function,
		Tanh_derivative_function,
		linear_derivative_function,
		gelu_derivative_function,
		swish_derivative_function
	};
	return map[function];
}