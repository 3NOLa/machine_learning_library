from python_binding.tasks import ffi, lib
from neuron import *
import Tensor


class Layer:
    def __init__(self, c_layer, type, neuron_amount : int,input_dim : int, activation_type=None):
        self.layer_type_ptr = c_layer
        self.type = type
        self.input_dim = input_dim
        self.activation_function = lib.LINEAR if activation_type is None else activation_type
        self.neuron_amount = neuron_amount
        self.layer_ptr = lib.general_layer_Initialize(self.type,self.neuron_amount,self.input_dim,self.activation_function)
        self.py_neurons = []

    def layer_forward(self, input:Tensor):
        raise NotImplementedError

    def layer_backward(self, input_gradients:Tensor,lr:float):
        raise NotImplementedError

    def get_neuron(self, index: int):
        return self.py_neurons[index]

    def __exit__(self):
        lib.layer_free(self.layer_ptr)


class DenseLayer(Layer):
    def __init__(self, neuron_amount: int, input_dim: int, activation_type=None):
        layer_ptr = lib.layer_create(neuron_amount,input_dim, activation_type)
        super().__init__(layer_ptr, lib.LAYER_DENSE, neuron_amount, input_dim, activation_type)
        self.py_neurons = [
            DenseNeuron(
                input_size=input_dim,
                c_neuron=self.layer_type_ptr.neurons[i],
                activation_type=activation_type
            )
            for i in range(neuron_amount)
        ]

    def layer_forward(self, input:Tensor):
        return lib.layer_forward(self.layer_type_ptr, input.CTensor)

    def layer_backward(self, input_gradients:Tensor,lr:float):
        return lib.layer_backward(self.layer_type_ptr, input_gradients.CTensor, lr)


class RnnLayer(Layer):
    def __init__(self, neuron_amount: int, input_dim: int, activation_type=1):
        layer_ptr = lib.rnn_layer_create(neuron_amount,input_dim, activation_type)
        super().__init__(layer_ptr, lib.LAYER_RNN, neuron_amount, input_dim, activation_type)
        self.py_neurons = [
            RnnNeuron(
                input_size=input_dim,
                c_neuron=self.layer_type_ptr.neurons[i],
                activation_type=activation_type
            )
            for i in range(neuron_amount)
        ]

    def layer_forward(self, input:Tensor):
        return lib.rnn_layer_forward(self.layer_type_ptr, input.CTensor)

    def layer_backward(self, input_gradients:Tensor,lr:float):
        return lib.rnn_layer_backward(self.layer_type_ptr, input_gradients.CTensor, lr)


class LstmLayer(Layer):
    def __init__(self, neuron_amount: int, input_dim: int, activation_type=None):
        layer_ptr = lib.lstm_layer_create(neuron_amount,input_dim, activation_type)
        super().__init__(layer_ptr, lib.LAYER_LSTM, neuron_amount, input_dim, activation_type)
        self.py_neurons = [
            LstmNeuron(
                input_size=input_dim,
                c_neuron=self.layer_type_ptr.neurons[i],
                activation_type=activation_type
            )
            for i in range(neuron_amount)
        ]

    def layer_forward(self, input:Tensor):
        return lib.lstm_layer_forward(self.layer_type_ptr, input.CTensor)

    def layer_backward(self, input_gradients:Tensor,lr:float):
        return lib.lstm_layer_backward(self.layer_type_ptr, input_gradients.CTensor, lr)

