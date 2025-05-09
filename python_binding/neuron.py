from python_binding.tasks import ffi, lib
import Tensor


class Neuron:
    def __init__(self, type, neuron, activation_function):
        self.type = type
        self.neuron = neuron
        self.activation_function = activation_function

    def activate_neuron(self, input):
        raise NotImplementedError

    def backward_neuron(self):
        raise NotImplementedError


class DenseNeuron(Neuron):
    def __init__(self, input_size, activation_type=None, c_neuron=None):
        type = lib.LAYER_DENSE
        activation_function = lib.LINEAR if activation_type is None else activation_type
        neuron = c_neuron if c_neuron is not None and ffi.typeof(c_neuron).cname == "neuron *" else lib.neuron_create(input_size, activation_function)

        super().__init__(type,neuron,activation_function)

    def activate_neuron(self, input : Tensor):
        return lib.neuron_activation(input.CTensor, self.neuron)

    def backward_neuron(self):
        return lib.neuron_backward()

    def __exit__(self, exc_type, exc_val, exc_tb):
        lib.neuron_free(self.neuron)


class RnnNeuron(Neuron):
    def __init__(self, input_size, activation_type=None, c_neuron=None):
        type = lib.LAYER_RNN
        activation_function = lib.LINEAR if activation_type is None else activation_type
        neuron = c_neuron if c_neuron is not None and ffi.typeof(c_neuron).cname == "rnn_neuron *" else lib.rnn_neuron_create(input_size, activation_function)

        super().__init__(type,neuron,activation_function)

    def activate_neuron(self, input : Tensor):
        return lib.rnn_neuron_activation(input.CTensor, self.neuron)

    def backward_neuron(self):
        return lib.rnn_neuron_backward()

    def __exit__(self, exc_type, exc_val, exc_tb):
        lib.rnn_neuron_free(self.neuron)


class LstmNeuron(Neuron):
    def __init__(self, input_size, activation_type=None, c_neuron=None):
        type = lib.LAYER_LSTM
        activation_function = lib.LINEAR if activation_type is None else activation_type
        neuron = c_neuron if c_neuron is not None and ffi.typeof(c_neuron).cname == "lstm_neuron *" else lib.lstm_neuron_create(input_size,activation_function)

        super().__init__(type,neuron,activation_function)

    def activate_neuron(self, input : Tensor):
        return lib.lstm_neuron_activation(input.CTensor, self.neuron)

    def backward_neuron(self):
        return lib.lstm_neuron_backward()

    def __exit__(self, exc_type, exc_val, exc_tb):
        lib.lstm_neuron_free(self.neuron)
