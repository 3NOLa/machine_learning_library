#include "tensor.h"
#include "neuron.h"
#include "rnn_neuron.h"
#include "lstm_neuron.h"
#include "weights_initialization.h"
#include <time.h>

Initializer* initializer_random_normal(float mean, float stddev) {
	srand(time(NULL));
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_random_normal\n");
		return NULL;
	}

	init->type = RandomNormal;
	init->args.random.mean = mean;
	init->args.random.stddev = stddev;
	init->tensor_init = random_normal_tensor_init;
	init->float_init = random_normal_float_init;
	return init;
}

Initializer* initializer_random_uniform(float min, float max) {
	srand(time(NULL));
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_random_uniform\n");
		return NULL;
	}

	init->type = RandomUniform;
	init->args.random.mean = min;
	init->args.random.stddev = max;
	init->tensor_init = random_uniform_tensor_init;
	init->float_init = random_uniform_float_init;
	return init;
}


Initializer* initializer_xavier_normal(int fan_in, int fan_out) {
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_xavier_normal\n");
		return NULL;
	}

	init->type = XavierNormal;
	init->args.xavier.fan_in = fan_in;
	init->args.xavier.fan_out = fan_out;
	init->tensor_init = xavier_normal_tensor_init;
	init->float_init = xavier_normal_float_init;
	return init;
}

Initializer* initializer_xavier_uniform(int fan_in, int fan_out) {
	srand(time(NULL));
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_xavier_uniform\n");
		return NULL;
	}

	init->type = XavierUniform;
	init->args.xavier.fan_in = fan_in;
	init->args.xavier.fan_out = fan_out;
	init->tensor_init = xavier_uniform_tensor_init;
	init->float_init = xavier_uniform_float_init;
	return init;
}


Initializer* initializer_he_normal(int fan_in) {
	srand(time(NULL));
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_he_normal\n");
		return NULL;
	}

	init->type = HeNormal;
	init->args.he.fan_in = fan_in;
	init->tensor_init = he_normal_tensor_init;
	init->float_init = he_normal_float_init;
	return init;
}

Initializer* initializer_he_uniform(int fan_in) {
	srand(time(NULL));
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_he_uniform\n");
		return NULL;
	}

	init->type = HeUniform;
	init->args.he.fan_in = fan_in;
	init->tensor_init = he_uniform_tensor_init;
	init->float_init = he_uniform_float_init;
	return init;
}

Initializer* initializer_lecun_normal(int fan_in) {
	srand(time(NULL));
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_lecun_normal\n");
		return NULL;
	}

	init->type = LeCunNormal;
	init->args.lecun.fan_in = fan_in;
	init->tensor_init = lecun_normal_tensor_init;
	init->float_init = lecun_normal_float_init;
	return init;
}

Initializer* initializer_lecun_uniform(int fan_in) {
	srand(time(NULL));
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_lecun_uniform\n");
		return NULL;
	}
	init->type = LeCunUniform;
	init->args.lecun.fan_in = fan_in;
	init->tensor_init = lecun_uniform_tensor_init;
	init->float_init = lecun_uniform_float_init;
	return init;
}

Initializer* initializer_orthogonal(float gain, int rows, int cols) {
	srand(time(NULL));
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_orthogonal\n");
		return NULL;
	}

	init->type = Orthogonal;
	init->args.orth.cols = cols;
	init->args.orth.rows = rows;
	init->args.orth.gain = gain;
	init->tensor_init = orthogonal_tensor_init;
	init->float_init = orthogonal_float_init;
	return init;
}

Initializer* initializer_sparse(int sparsity_level, int nonzero_initializer) {
	srand(time(NULL));
	Initializer* init = (Initializer*)malloc(sizeof(Initializer));
	if (!init) {
		fprintf(stderr, "Error: Memory allocation failed for Initializer in initializer_sparse\n");
		return NULL;
	}

	init->type = Sparse;
	init->args.sparse.nonzero_initializer = nonzero_initializer;
	init->args.sparse.sparsity_level = sparsity_level;
	init->tensor_init = sparse_tensor_init;
	init->float_init = sparse_float_init;
	return init;
}

void random_normal_tensor_init(Tensor* data, Initializer* init) {
	float random_float;
	for (int i = 0; i < data->count; i++) {
		random_float = (float)rand() / (float)RAND_MAX;
		data->data[i] = random_float * (init->args.random.stddev - init->args.random.mean) + init->args.random.mean;
	}
}

void random_normal_float_init(float* data, Initializer* init) {
	float random_float = (float)rand() / (float)RAND_MAX;
	*data = random_float * (init->args.random.stddev - init->args.random.mean) + init->args.random.mean;
}

void random_uniform_tensor_init(Tensor* data, Initializer* init) {
	float random_float;
	for (int i = 0; i < data->count; i++) {
		random_float = (float)rand() / (float)RAND_MAX;
		data->data[i] = random_float * (2 * init->args.random.a) - init->args.random.a;
	}
}

void random_uniform_float_init(float* data, Initializer* init) {
	float random_float = (float)rand() / (float)RAND_MAX;
	*data = random_float * (2 * init->args.random.a) - init->args.random.a;
}

void xavier_normal_tensor_init(Tensor* data, Initializer* init) {
	float stddev, random_float;
	for (int i = 0; i < data->count; i++) {
		stddev = sqrtf(2.0 / (init->args.xavier.fan_in + init->args.xavier.fan_out));
		random_float = (float)rand() / (float)RAND_MAX;
		data->data[i] = random_float * stddev;
	}
}

void xavier_normal_float_init(float* data, Initializer* init) {
	float stddev = sqrtf(2.0 / (init->args.xavier.fan_in + init->args.xavier.fan_out));
	float random_float = (float)rand() / (float)RAND_MAX;
	*data = random_float * stddev;
}

void xavier_uniform_tensor_init(Tensor* data, Initializer* init) {
	float a, random_float;
	for (int i = 0; i < data->count; i++) {
		a = sqrtf(6.0 / (init->args.xavier.fan_in + init->args.xavier.fan_out));
		random_float = (float)rand() / (float)RAND_MAX;
		data->data[i] = random_float * (2 * a) - a;
	}
}

void xavier_uniform_float_init(float* data, Initializer* init) {
	float a = sqrtf(6.0 / (init->args.xavier.fan_in + init->args.xavier.fan_out));
	float random_float = (float)rand() / (float)RAND_MAX;
	*data = random_float * (2 * a) - a;
}

void he_normal_tensor_init(Tensor* data, Initializer* init) {
	float stddev, random_float;
	for (int i = 0; i < data->count; i++) {
		stddev = sqrtf(2.0 / init->args.he.fan_in);
		random_float = (float)rand() / (float)RAND_MAX;
		data->data[i] = random_float * stddev;
	}
}

void he_normal_float_init(float* data, Initializer* init) {
	float stddev = sqrtf(2.0 / init->args.xavier.fan_in);
	float random_float = (float)rand() / (float)RAND_MAX;
	*data = random_float * stddev;
}

void he_uniform_tensor_init(Tensor* data, Initializer* init) {
	float a, random_float;
	for (int i = 0; i < data->count; i++) {
		a = sqrtf(6.0 / init->args.he.fan_in);
		random_float = (float)rand() / (float)RAND_MAX;
		data->data[i] = random_float * (2 * a) - a;
	}
}

void he_uniform_float_init(float* data, Initializer* init) {
	float a = sqrtf(6.0 / init->args.he.fan_in);
	float random_float = (float)rand() / (float)RAND_MAX;
	*data = random_float * (2 * a) - a;
}

void lecun_normal_tensor_init(Tensor* data, Initializer* init) {
	float stddev, random_float;
	for (int i = 0; i < data->count; i++) {
		stddev = sqrtf(1.0 / init->args.lecun.fan_in);
		random_float = (float)rand() / (float)RAND_MAX;
		data->data[i] = random_float * stddev;
	}
}
void lecun_normal_float_init(float* data, Initializer* init) {
	float stddev = sqrtf(1.0 / init->args.lecun.fan_in);
	float random_float = (float)rand() / (float)RAND_MAX;
	*data = random_float * stddev;
}

void lecun_uniform_tensor_init(Tensor* data, Initializer* init) {
	float a, random_float;
	for (int i = 0; i < data->count; i++) {
		a = sqrtf(3.0 / init->args.lecun.fan_in);
		random_float = (float)rand() / (float)RAND_MAX;
		data->data[i] = random_float * (2 * a) - a;
	}
}

void lecun_uniform_float_init(float* data, Initializer* init) {
	float a = sqrtf(3.0 / init->args.lecun.fan_in);
	float random_float = (float)rand() / (float)RAND_MAX;
	*data = random_float * (2 * a) - a;
}

void neuron_opt_init(neuron* n, Initializer* init) {
	init->float_init(&n->bias, init);
	init->tensor_init(n->weights, init);
}

void rnn_neuron_opt_init(rnn_neuron* rn, Initializer* init) {
	neuron_opt_init(rn->n, init);
	init->float_init(&rn->recurrent_weights, init);
}

void lstm_neuron_opt_init(lstm_neuron* ln, Initializer* init) {
	rnn_neuron_opt_init(ln->f_g,init);
	rnn_neuron_opt_init(ln->i_g_p, init);
	rnn_neuron_opt_init(ln->i_g_r, init);
	rnn_neuron_opt_init(ln->o_g_r, init);
}


//need to implenet this 
void orthogonal_tensor_init(Tensor* data, Initializer* init) {
	return;
}

void orthogonal_float_init(float* data, Initializer* init) {
	return;
}

void sparse_tensor_init(Tensor* data, Initializer* init) {
	return;
}
void sparse_float_init(float* data, Initializer* init) {
	return;
}
