#include "active_functions.h"

void RELu_layer(Matrix* mat)
{
	for (int i = 0; i < mat->cols * mat->rows; i++)
		mat->data[i] = (mat->data[i] > 0) ? mat->data[i] : 0;
}

void Sigmoid_layer(Matrix* mat)
{
	for (int i = 0; i < mat->cols * mat->rows; i++)
		mat->data[i] = 1 / (1 + exp(-mat->data[i])); // exp = e ^ -mat->data[i])
}

void Tanh_layer(Matrix* mat)
{
	for (int i = 0; i < mat->cols * mat->rows; i++)
	{
		double x = exp(mat->data[i]);
		double y = exp(-mat->data[i]);
		mat->data[i] = (x - y) / (x+y);
	}
		
}

double RELu(double value)
{
	return (value > 0) ? value : 0;
}

double Sigmoid(double value)
{
	return 1 / (1 + exp(-value));
}

double Tanh(double value)
{
	double x = exp(value);
	double y = exp(-value);
	return (x - y) / (x + y);
}

void (*ActivationTypeMap(ActivationType function))(double value)
{
	static void (*map[])(double) = { RELu, Sigmoid, Tanh };
	return map[function];
}