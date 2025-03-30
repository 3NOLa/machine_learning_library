#include "active_functions.h"

double RELu_function(double value)
{
	return (value > 0) ? value : 0;
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

void (*ActivationTypeMap(ActivationType function))(double)
{
	static void (*map[])(double) = { RELu_function, Sigmoid_function, Tanh_function };
	return map[function];
}

double RELu_derivative_function(double value)
{
	return (value > 0) ? 1 : 0;
}

double Sigmoid_derivative_function(double sigmoid)
{
	return sigmoid * (1 - sigmoid); 
}

double Tanh_derivative_function(double Tanh)
{
	return 1 - Tanh * Tanh; // 1 - tanh^2
}

void (*ActivationTypeDerivativeMap(ActivationType function))(double)
{
	static void (*map[])(double) = { RELu_derivative_function, Sigmoid_derivative_function, Tanh_derivative_function };
	return map[function];
}