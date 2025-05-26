typedef struct {
	int dims;
	int* shape; //shaoes of the dims
	int* strides; //amount of bytes i need to get to next dim;
	int count; //amount of elemnts
	float* data;
} Tensor;

Tensor* tensor_create(int dims, int* shape);
Tensor* tensor_create_flatten(int dims, int* shape,float* flatten, int count);
Tensor* tensor_zero_create(int dims, int* shape);
Tensor* tensor_random_create(int dims, int* shape);
Tensor* tensor_identity_create(int row);
int tensor_add_row(Tensor* t);

void tensor_free(Tensor* t);
int tensor_copy(Tensor* dest, Tensor* src);
void tensor_fill(Tensor* t,float value);

// Access functions
int tensor_get_index(Tensor* t, int* indices);
float  tensor_get_element(Tensor* t, int* indices);
float  tensor_get_element_by_index(Tensor* t, int index);
void tensor_set(Tensor* t, int* indices, float  value);
void tensor_set_by_index(Tensor* t, int index, float  value);

// Dimension manipulation
Tensor* tensor_reshape(Tensor* t, int dims, int* shape);
Tensor* tensor_flatten(Tensor* t); // Convert to 1D tensor
Tensor* tensor_slice_range(Tensor* t, int start, int end);
void tensor_squeeze(Tensor* t);
Tensor* tensor_get_row(Tensor* t, int row);
Tensor* tensor_get_col(Tensor* t, int col);

// Math operations
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_subtract(Tensor* a, Tensor* b);
Tensor* tensor_multiply(Tensor* a, Tensor* b); // Element-wise multiplication
Tensor* tensor_dot(Tensor* a, Tensor* b);      // Matrix multiplication when applicable
Tensor* tensor_add_scalar(Tensor* t, float  scalar);
Tensor* tensor_multiply_scalar(Tensor* t, float  scalar);
float  tensor_sum(Tensor* t);
float  tensor_mean(Tensor* t);

// In-place operations (to minimize memory allocations)
void tensor_add_inplace(Tensor* target, Tensor* other);
void tensor_add_more_inplace(Tensor* target, Tensor* others[], int amount);
void tensor_subtract_inplace(Tensor* target, Tensor* other);
void tensor_multiply_inplace(Tensor* target, Tensor* other);
void tensor_add_scalar_inplace(Tensor* target, float  scalar);
void tensor_multiply_scalar_inplace(Tensor* target, float  scalar);

// Print tensor
void tensor_print(Tensor* t);


typedef enum {
	SGD, // SCHOLAR GRADINET DECENT
	SGDM,// SGD WITH MOMENTUM
	NESTEROV, // SGD WITH Nesterov Momentum
	ADAM,
	RMSPROP,
}OptimizerType;

typedef struct {
	Tensor* velocity;
	float fvelocity;
	float momentum; // beta
} MomentumState;

typedef struct {
	Tensor* velocity;
	float fvelocity;
	float momentum;
}NesterovMomentumState;

typedef struct {
	Tensor* avg_sq_grad;
	float favg_sq_grad;
	float decay;
	float epsilon;
} RMSPropState;

typedef struct {
	Tensor* m;  // 1st moment (mean)
	Tensor* v;  // 2nd moment (variance)
	int t;      // timestep
	float beta1;
	float beta2;
	float epsilon;
} AdamState;

typedef union {
	MomentumState momentum;
	NesterovMomentumState nesterov;
	RMSPropState rmsprop;
	AdamState adam;
}OptimizerArgs;

typedef struct optimizer{
    OptimizerType type;
	OptimizerArgs args;
	void (*tensor_update)(Tensor*, Tensor*, float, OptimizerArgs*);
	void (*float_update)(float*, float*, float, OptimizerArgs*);
} optimizer;


void optimizer_set(optimizer* op, OptimizerType type);

void sgd_tensor_update(Tensor* data, Tensor* grad,float lr, OptimizerArgs* args);
void sgd_float_update(float* data, float* grad, float lr, OptimizerArgs* args);

void sgdm_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args);
void sgdm_float_update(float* data, float* grad, float lr, OptimizerArgs* args);

void nesterov_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args);
void nesterov_float_update(float* data, float* grad, float lr, OptimizerArgs* args);

void adam_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args);
void adam_float_update(float* data, float* grad, float lr, OptimizerArgs* args);

void rmsprop_tensor_update(Tensor* data, Tensor* grad, float lr, OptimizerArgs* args);
void rmsprop_float_update(float* data, float* grad, float lr, OptimizerArgs* args);


typedef enum {
	RELU,
	LEAKY_RELU,
	SIGMOID,
	TANH,
	LINEAR,
	GELU,
	SWISH
}ActivationType;

typedef struct neuron {
	Tensor* weights;
    	Tensor* grad_weights;
    	float bias;
   	float grad_bias;
	Tensor* input;
	float  pre_activation;
	float  output;
	ActivationType Activation;
	float  (*ActivationFunc)(float  value);
	float  (*ActivationderivativeFunc)(struct neuron*);
    	optimizer* opt;
} neuron;

neuron* neuron_create(int weightslength, ActivationType func);
void neuron_set_ActivationType(neuron* n, ActivationType Activation);
float  neuron_activation(Tensor* input, neuron* n);
void neuron_backward(float  derivative, neuron* n, Tensor* output_gradients);
void neuron_update(neuron* n, float learning_rate);
void neuron_zero_grad(neuron* n);
void neuron_free(neuron* n);
void neuron_opt_update(neuron* n, optimizer* opt, float lr);

float  RELu_function(float  value);
float  RELu_derivative_function(neuron* n); 

float  leaky_RELu_function(float  value);
float  leaky_RELu_derivative_function(neuron* n);

float  Sigmoid_function(float  value);
float  Sigmoid_derivative_function(neuron* n);

float  Tanh_function(float  value);
float  Tanh_derivative_function(neuron* n);

float  linear_function(float  value);
float  linear_derivative_function(neuron* n);

float  gelu_function(float  value);
float  gelu_derivative_function(neuron* n);

float  swish_function(float  value);
float  swish_derivative_function(neuron* n);

float  (*ActivationTypeMap(ActivationType function))(float);
float  (*ActivationTypeDerivativeMap(ActivationType function))(neuron*);


typedef enum {
	LAYER_DENSE,
	LAYER_RNN,
	LAYER_LSTM,
	// LAYER_CONV
} LayerType;


 typedef struct Layer {
    LayerType type;
    int neuronAmount;
    void* params;  
    Tensor* (*forward)(struct Layer* layer, Tensor* input);
    Tensor* (*backward)(struct Layer* layer, Tensor* grad);
    void (*update)(struct Layer* layer, float  learning_rate);
    void (*free)(struct Layer* layer);
    void (*zero_grad)(struct Layer* layer);
    //rnn only
    void (*reset_state)(struct layer* base_layer);
}layer;

layer* general_layer_Initialize(LayerType type, int neuronAmount, int neuronDim, ActivationType Activationfunc);
void general_layer_free(layer* base_layer);
Tensor* get_layer_output(layer* base_layer);
void set_layer_output(layer* base_layer, Tensor* output);
void set_layer_optimizer(layer* base_layer, OptimizerType type);
void set_layer_output(layer* base_layer, Tensor* output);

Tensor* wrapper_rnn_forward(layer* base_layer, Tensor* input);
Tensor* wrapper_rnn_backward(layer* base_layer, Tensor* grad);
void wrapper_rnn_update(layer* base_layer, float lr);
void wrapper_rnn_zero_grad(layer* base_layer);
void wrapper_rnn_reset_state(layer* base_layer);

Tensor* wrapper_dense_forward(layer* base_layer, Tensor* input);
Tensor* wrapper_dense_backward(layer* base_layer, Tensor* gra);
void wrapper_dense_update(layer* base_layer, float lr);
void wrapper_dense_zero_grad(layer* base_layer);

Tensor* wrapper_lstm_forward(layer* base_layer, Tensor* input);
Tensor* wrapper_lstm_backward(layer* base_layer, Tensor* grad);
void wrapper_lstm_update(layer* base_layer, float lr);
void wrapper_lstm_zero_grad(layer* base_layer);
void wrapper_lstm_reset_state(layer* base_layer);

typedef enum {
	MSE,
	MAE,
	Binary_Cross_Entropy,
	Categorical_Cross_Entropy,
	Huber_Loss
}LossType;


typedef struct {
	int layerAmount;
	layer** layers;
	int* layersSize;
	float  learnningRate;
	LossType lossFunction;
	float  (*LossFuntionPointer)(struct network*, Tensor*);
	Tensor* (*LossDerivativePointer)(struct network*, Tensor*);
	float  (*train)(struct network*, Tensor*, Tensor*);
	int input_dims;
	int* input_shape;
	LayerType type;
	OptimizerType otype;
}network;

network* network_create(int layerAmount, int* layersSize, int input_dims, int* input_shape, ActivationType* activations, float  learnningRate, LossType lossFunction, LayerType type);
network* network_create_empty();
int add_created_layer(network* net, layer* l);
int add_layer(network* net, int layerSize, ActivationType Activationfunc, int input_dim);// add input layer if first layer otherwise put 0
int set_loss_function(network* net, LossType lossFunction);
void set_network_optimizer(network* net, OptimizerType type);
void network_free(network* net);
void network_train_type(network* net);

Tensor* forwardPropagation(network* net, Tensor* data);
int backpropagation(network* net, Tensor* predictions, Tensor* targets);
void network_update(network* net);
void network_zero_grad(network* net);
void network_reset_state(network* net);

float  train(network* net, Tensor* input, Tensor* target);
void network_training(network* net, Tensor* input, Tensor* target, int epcho, int batch_size);
float  rnn_train(network* net, Tensor* input, Tensor* target, int timestamps);

// implemented in loss_Functions.c
float  squared_error_net(network* net, Tensor* y_real);
Tensor* derivative_squared_error_net(network* net, Tensor* y_real);

float  absolute_error_net(network* net, Tensor* y_real);
Tensor* derivative_absolute_error_net(network* net, Tensor* y_real);

float  absolute_error_net(network* net, Tensor* y_real);
Tensor* derivative_absolute_error_net(network* net, Tensor* y_real);

float  Categorical_Cross_Entropy_net(network* net, Tensor* y_real);
Tensor* derivative_Categorical_Cross_Entropy_net(network* net, Tensor* y_real);

float  (*LossTypeMap(LossType function))(network*, Tensor*);
Tensor* (*LossTypeDerivativeMap(LossType function))(network*, Tensor*);

float loss_active_function(LossType function, Tensor* y_pred, Tensor* y_real);
Tensor* loss_derivative_active_function(LossType function, Tensor* y_pred, Tensor* y_real);

typedef struct {
	int num_classes;
	Tensor* one_hot_encode;
	char** class_names;
	network* net;
} ClassificationNetwork;

ClassificationNetwork* ClassificationNetwork_create(int layerAmount, int* layersSize, int* input_shape, int input_dim, ActivationType* activations, float  learnningRate, LossType lossFunction, char** class_names, int num_classes, Tensor* classes, LayerType type);
ClassificationNetwork* ClassificationNetwork_create_net(network* net, char** class_names, int num_classes, Tensor* classes);
Tensor* one_hot_encode(int num_classes);
void classification_info_free(ClassificationNetwork* info);

int get_predicted_class(Tensor* network_output);
void classification_network_training(ClassificationNetwork* Cnet);


typedef struct {
	int neuronAmount;
	Tensor* output;
	neuron** neurons;
	ActivationType Activationenum;
}dense_layer;

dense_layer* layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc);
void layer_removeLastNeuron(dense_layer* l);
void layer_addNeuron(dense_layer* l);
void layer_set_neuronAmount(dense_layer* l, int neuronAmount);
void layer_set_activtion(dense_layer* l, ActivationType Activationfunc);
Tensor* layer_forward(dense_layer* l, Tensor* input);
Tensor* layer_backward(dense_layer* l, Tensor* input_gradients);
void dense_layer_update(dense_layer* layer, float learning_rate);
void dense_layer_zero_grad(dense_layer* dl);
void layer_free(dense_layer* l);





typedef struct rnn_neuron {
	neuron* n;
	float recurrent_weights;
    	float grad_recurrent_weights;
    	float hidden_state;
   	float grad_hidden_state;
	Tensor* input_history[128];
	float  hidden_state_history[128];
	int timestamp;
	optimizer* opt;
} rnn_neuron;

rnn_neuron* rnn_neuron_create(int weightslength, ActivationType func);
void rnn_neuron_set_ActivationType(rnn_neuron* rn, ActivationType Activation);
float rnn_neuron_activation(Tensor* input, rnn_neuron* rn);
void rnn_neuron_backward(float  output_gradient, rnn_neuron* rn, Tensor* input_grads);
void rnn_neuron_update(rnn_neuron* rn, float rl);
void rnn_neuron_zero_grad(rnn_neuron* rn);
void rnn_neuron_free(rnn_neuron* rn);
void rnn_neuron_opt_update(rnn_neuron* rn, optimizer* opt, float lr);

typedef struct lstm_neuron {
	float  short_memory;//cell state
	float  long_memory;
	Tensor* input_history[128];
	float  short_memory_history[128];
	float  long_memory_history[128];//cell state history
	int timestamp;

	rnn_neuron* i_g_r; //input_gate_remember
	rnn_neuron* i_g_p; //input_gate_potinal also known as candidate cell
	rnn_neuron* o_g_r; //output_gate_remember 
	rnn_neuron* f_g; //forget_gate
   	optimizer* opt;
} lstm_neuron;

lstm_neuron* lstm_neuron_create(int weightslength, ActivationType func);
float  lstm_neuron_activation(Tensor* input, lstm_neuron* ln);
void lstm_neuron_backward(float  derivative, lstm_neuron* ln, Tensor* input_gradients);
void lstm_neuron_update(lstm_neuron* ln, float rl);
void lstm_neuron_zero_grad(lstm_neuron* ln);
void lstm_neuron_free(lstm_neuron* ln);
void lstm_neuron_opt_update(lstm_neuron* ln, optimizer* opt, float lr);

typedef struct {
	int neuronAmount;
	Tensor* output;
	rnn_neuron** neurons;
	ActivationType Activationenum;
	int sequence_length; // =t = timestamp
}rnn_layer;

rnn_layer* rnn_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc);
Tensor* rnn_layer_forward(rnn_layer* rl, Tensor* input);
Tensor* rnn_layer_backward(rnn_layer* rl, Tensor* output_gradients);
void rnn_layer_update(rnn_layer* rl, float lr);
void rnn_layer_zero_grad(rnn_layer* rl);
void rnn_layer_reset_state(rnn_layer* rl);
void rnn_layer_free(rnn_layer* rl);

typedef struct {
	int neuronAmount;
	Tensor* output;
	lstm_neuron** neurons;
	ActivationType Activationenum;
	int sequence_length;
}lstm_layer;

lstm_layer* lstm_layer_create(int neuronAmount, int neuronDim, ActivationType Activationfunc);
Tensor* lstm_layer_forward(lstm_layer* ll, Tensor* input);
Tensor* lstm_layer_backward(lstm_layer* ll, Tensor* output_gradients);
void lstm_layer_update(lstm_layer* ll, float lr);
void lstm_layer_zero_grad(lstm_layer* ll);
void lstm_layer_reset_state(lstm_layer* ll);
void lstm_layer_free(lstm_layer* ll);

char** tokeknize(const char* text, int* token_count);
void to_lowercase(char* str);
char* remove_punctuation(const char* input);