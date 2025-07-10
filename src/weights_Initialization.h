#pragma once
#include "export.h"

typedef struct rnn_neuron rnn_neuron;
typedef struct lstm_neuron lstm_neuron;
typedef struct Tensor;
typedef struct neuron;

EXPORT typedef enum initializerType {
	RandomNormal,
	RandomUniform,
	XavierNormal,
	XavierUniform,
	HeNormal,
	HeUniform,
	LeCunNormal,
	LeCunUniform,
	Orthogonal,
	Sparse
}initializerType;

EXPORT typedef struct {
	float a;
	int mean;
	int stddev;
} RandomaArgs;

EXPORT typedef struct {
	int fan_in; // weight amount
	int fan_out; // neuron amount
}XavierArgs;

EXPORT typedef struct {
	int fan_in; // weight amount
} HeArgs;

EXPORT typedef struct {
	int fan_in; // weight amount
} LeCunArgs;

EXPORT typedef struct {
	float gain;
	int rows;
	int cols; //shape
} OrthogonalArgs;

EXPORT typedef struct {
	int sparsity_level;
	int nonzero_initializer;
} SparseArgs;

EXPORT typedef union {
	RandomaArgs random;
	XavierArgs xavier;
	HeArgs he;
	LeCunArgs lecun;
	OrthogonalArgs orth;
	SparseArgs sparse;
}InitializerArgs;

EXPORT typedef struct Initializer {
	initializerType type;
	InitializerArgs args;
	void (*tensor_init)(Tensor*, struct Initializer*);
	void (*float_init)(float*, struct Initializer*);
} Initializer;

EXPORT Initializer* initializer_random_normal(float mean, float stddev);
EXPORT Initializer* initializer_random_uniform(float min, float max);

EXPORT Initializer* initializer_xavier_normal(int fan_in, int fan_out);
EXPORT Initializer* initializer_xavier_uniform(int fan_in, int fan_out);

EXPORT Initializer* initializer_he_normal(int fan_in);
EXPORT Initializer* initializer_he_uniform(int fan_in);

EXPORT Initializer* initializer_lecun_normal(int fan_in);
EXPORT Initializer* initializer_lecun_uniform(int fan_in);

EXPORT Initializer* initializer_orthogonal(float gain, int rows, int cols);
EXPORT Initializer* initializer_sparse(int sparsity_level, int nonzero_initializer);

// Initialization logic for tensors and scalars
EXPORT void random_normal_tensor_init(Tensor* data, Initializer* init);
EXPORT void random_normal_float_init(float* data, Initializer* init);
EXPORT void random_uniform_tensor_init(Tensor* data, Initializer* init);
EXPORT void random_uniform_float_init(float* data, Initializer* init);

EXPORT void xavier_normal_tensor_init(Tensor* data, Initializer* init);
EXPORT void xavier_normal_float_init(float* data, Initializer* init);
EXPORT void xavier_uniform_tensor_init(Tensor* data, Initializer* init);
EXPORT void xavier_uniform_float_init(float* data, Initializer* init);

EXPORT void he_normal_tensor_init(Tensor* data, Initializer* init);
EXPORT void he_normal_float_init(float* data, Initializer* init);
EXPORT void he_uniform_tensor_init(Tensor* data, Initializer* init);
EXPORT void he_uniform_float_init(float* data, Initializer* init);

EXPORT void lecun_normal_tensor_init(Tensor* data, Initializer* init);
EXPORT void lecun_normal_float_init(float* data, Initializer* init);
EXPORT void lecun_uniform_tensor_init(Tensor* data, Initializer* init);
EXPORT void lecun_uniform_float_init(float* data, Initializer* init);

EXPORT void orthogonal_tensor_init(Tensor* data, Initializer* init);
EXPORT void orthogonal_float_init(float* data, Initializer* init);

EXPORT void sparse_tensor_init(Tensor* data, Initializer* init);
EXPORT void sparse_float_init(float* data, Initializer* init);

EXPORT void neuron_opt_init(neuron* n, Initializer* init);
EXPORT void rnn_neuron_opt_init(rnn_neuron* rn, Initializer* init);
EXPORT void lstm_neuron_opt_init(lstm_neuron* ln, Initializer* init);
