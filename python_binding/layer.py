from python_binding.tasks import ffi, lib
from neuron import *
from Tensor import Tensor


class Layer:
    def __init__(self, type, neuron_amount : int,input_dim : int, activation_type=None):
        self.layer_type_ptr = None
        self.type = type
        self.input_dim = input_dim
        self.activation_function = lib.LINEAR if activation_type is None else activation_type
        self.neuron_amount = neuron_amount
        self.layer_ptr = lib.general_layer_Initialize(self.type,self.neuron_amount,self.input_dim,self.activation_function)
        self.py_neurons = []

    def layer_forward(self, input:Tensor):
        return Tensor.c_to_tensor(lib.layer_ptr.forward(self.layer_ptr,input))

    def layer_backward(self, input_gradients:Tensor,lr:float):
        return Tensor.c_to_tensor(lib.layer_ptr.backward(self.layer_ptr,input_gradients,lr))

    def get_neuron(self, index: int):
        return self.py_neurons[index]

    def __exit__(self):
        lib.layer_free(self.layer_ptr)


class DenseLayer(Layer):
    def __init__(self, neuron_amount: int, input_dim: int, activation_type=None):
        super().__init__(lib.LAYER_DENSE, neuron_amount, input_dim, activation_type)
        self.layer_type_ptr = lib.layer_create(neuron_amount,input_dim, self.activation_function)

        self.py_neurons = [
            DenseNeuron(
                input_size=input_dim,
                c_neuron=self.layer_type_ptr.neurons[i],
                activation_type=activation_type
            )
            for i in range(neuron_amount)
        ]

    def layer_forward(self, input:Tensor):
        return Tensor.c_to_tensor(lib.layer_forward(self.layer_type_ptr, input.c_tensor))

    def layer_backward(self, input_gradients:Tensor,lr:float):
        return Tensor.c_to_tensor(lib.layer_backward(self.layer_type_ptr, input_gradients.c_tensor, lr))


class RnnLayer(Layer):
    def __init__(self, neuron_amount: int, input_dim: int, activation_type=1):
        super().__init__(lib.LAYER_RNN, neuron_amount, input_dim, activation_type)
        self.layer_type_ptr = lib.rnn_layer_create(neuron_amount,input_dim, self.activation_function)

        self.py_neurons = [
            RnnNeuron(
                input_size=input_dim,
                c_neuron=self.layer_type_ptr.neurons[i],
                activation_type=activation_type
            )
            for i in range(neuron_amount)
        ]

    def layer_forward(self, input:Tensor, timestamps=1):
        t_outputs = []
        for t in range(timestamps): #need tp chamge for eatch timesatmp a slice of the input
            t_outputs.append(Tensor.c_to_tensor(lib.rnn_layer_forward(self.layer_type_ptr, input.c_tensor)))
        return t_outputs

    def layer_backward(self, input_gradients:Tensor,lr:float):
        return Tensor.c_to_tensor(lib.rnn_layer_backward(self.layer_type_ptr, input_gradients.c_tensor, lr))


class LstmLayer(Layer):
    def __init__(self, neuron_amount: int, input_dim: int, activation_type=None):
        super().__init__(lib.LAYER_LSTM, neuron_amount, input_dim, activation_type)
        self.layer_type_ptr = lib.lstm_layer_create(neuron_amount,input_dim, self.activation_function)

        self.py_neurons = [
            LstmNeuron(
                input_size=input_dim,
                c_neuron=self.layer_type_ptr.neurons[i],
                activation_type=activation_type
            )
            for i in range(neuron_amount)
        ]

    def layer_forward(self, input:Tensor, timestamps=1):
        t_outputs = []
        for i in range(timestamps):#need tp chamge for eatch timesatmp a slice of the input
            t_outputs.append(Tensor.c_to_tensor(lib.lstm_layer_forward(self.layer_type_ptr, input.c_tensor)))
        return t_outputs

    def layer_backward(self, input_gradients:Tensor,lr:float):
        return Tensor.c_to_tensor(lib.lstm_layer_backward(self.layer_type_ptr, input_gradients.c_tensor, lr))

