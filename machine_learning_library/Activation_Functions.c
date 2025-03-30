#include "active_functions.h"
#include "neuron.h"

double RELu_function(double value)
{
	return (value > 0) ? value : 0;
}

double leaky_RELu_function(double value)
{
	return (value > 0) ? value : 0.01 * value;
}

double Sigmoid_function(double value)
{
	return 1 / (1 + exp(-value)); //s(x) = 1 / (1 + e^-(x))
}

double Tanh_function(double value)
{
	double x = exp(value);
	double y = exp(-value);
	return (x - y) / (x + y);
}

double linear_function(double value)
{
	return value;
}

double gelu_function(double value)
{
	double s = sqrt(2 / M_PI) * (value + 0.044715 * value * value * value);
	return 0.5 * value * (1 + Tanh_function(s));
}

double swish_function(double value)
{
	return value * Sigmoid_function(value);
}

double (*ActivationTypeMap(ActivationType function))(double)
{
	static void (*map[])(double) = { RELu_function, 
		leaky_RELu_function, 
		Sigmoid_function, 
		Tanh_function,
		linear_function,gelu_function };
	return map[function];
}

double RELu_derivative_function(neuron* n)
{
	return (n->output > 0) ? 1 : 0;
}

double leaky_RELu_derivative_function(neuron* n)
{
	return (n->output > 0) ? 1 : 0.01;
}

double Sigmoid_derivative_function(neuron* n)
{
	return n->output * (1 - n->output); 
}

double Tanh_derivative_function(neuron* n)
{
	return 1 - n->output * n->output; // 1 - tanh^2
}

double linear_derivative_function(neuron* n)
{
	return 1;
}
	
double gelu_derivative_function(neuron* n)
{
	double tanh = (2 * n->output) / n->pre_activation - 1;
	return 0.5 * (1 + tanh) + 0.5 * n->pre_activation * (1 - tanh * tanh) * sqrt(2 / M_PI) * (1 + 3 * 0.044715 * n->pre_activation * n->pre_activation);
}

double swish_derivative_function(neuron* n)
{
	return Sigmoid_function(n->pre_activation) + Sigmoid_derivative_function(n) * n->pre_activation;
}

double (*ActivationTypeDerivativeMap(ActivationType function))(neuron*)
{
	static double (*map[])(neuron*) = {
		RELu_derivative_function,
		leaky_RELu_derivative_function,
		Sigmoid_derivative_function,
		Tanh_derivative_function,
		linear_derivative_function,
		gelu_derivative_function
	};
	return map[function];
}