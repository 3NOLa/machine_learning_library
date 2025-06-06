from python_binding.tasks import ffi, lib
from MyTensor import Tensor
from py_enums import *


class Neuron:
    def __init__(self, type, neuron, activation_function: ActivationType):
        self.type = type
        self.neuron = neuron
        self.activation_function = activation_function

    def activate_neuron(self, input):
        raise NotImplementedError("Subclasses must implement activate_neuron")

    def backward_neuron(self):
        raise NotImplementedError("Subclasses must implement backward_neuron")


class DenseNeuron(Neuron):
    def __init__(self, input_size: int, activation_type=None, c_neuron=None):
        neuron_type = lib.LAYER_DENSE
        activation_function = lib.LINEAR if activation_type is None else activation_type

        if c_neuron is not None:
            # Use provided C neuron
            if c_neuron == ffi.NULL:
                raise ValueError("Received null C neuron pointer")
            neuron_ptr = c_neuron
        else:
            # Create new C neuron
            neuron_ptr = lib.neuron_create(input_size, activation_function)
            if neuron_ptr == ffi.NULL:
                raise RuntimeError("Failed to create dense neuron")

        super().__init__(neuron_type, neuron_ptr, activation_function)

    def activate_neuron(self, input_tensor: Tensor) -> float:
        if not isinstance(input_tensor, Tensor):
            raise TypeError("Input must be a Tensor object")

        try:
            result = lib.neuron_activation(input_tensor.c_tensor, self.neuron)
            return float(result)
        except Exception as e:
            raise RuntimeError(f"Dense neuron activation failed: {e}")

    def backward_neuron(self):
        try:
            return lib.neuron_backward()
        except Exception as e:
            raise RuntimeError(f"Dense neuron backward pass failed: {e}")

    def __del__(self):
        # Only free if we created the neuron ourselves (not provided via c_neuron)
        # This is tricky to track, so we'll be conservative and not free here
        # The layer should handle cleanup
        pass


class RnnNeuron(Neuron):
    def __init__(self, input_size: int, activation_type=None, c_neuron=None):
        neuron_type = lib.LAYER_RNN
        activation_function = lib.LINEAR if activation_type is None else activation_type

        if c_neuron is not None:
            # Use provided C neuron
            if c_neuron == ffi.NULL:
                raise ValueError("Received null RNN neuron pointer")
            neuron_ptr = c_neuron
        else:
            # Create new C neuron
            neuron_ptr = lib.rnn_neuron_create(input_size, activation_function)
            if neuron_ptr == ffi.NULL:
                raise RuntimeError("Failed to create RNN neuron")

        super().__init__(neuron_type, neuron_ptr, activation_function)

    def activate_neuron(self, input_tensor: Tensor) -> float:
        if not isinstance(input_tensor, Tensor):
            raise TypeError("Input must be a Tensor object")

        try:
            result = lib.rnn_neuron_activation(input_tensor.c_tensor, self.neuron)
            return float(result)
        except Exception as e:
            raise RuntimeError(f"RNN neuron activation failed: {e}")

    def backward_neuron(self):
        try:
            return lib.rnn_neuron_backward()
        except Exception as e:
            raise RuntimeError(f"RNN neuron backward pass failed: {e}")

    def __del__(self):
        # Conservative approach - let the layer handle cleanup
        pass


class LstmNeuron(Neuron):
    def __init__(self, input_size: int, activation_type=None, c_neuron=None):
        neuron_type = lib.LAYER_LSTM
        activation_function = lib.LINEAR if activation_type is None else activation_type

        if c_neuron is not None:
            # Use provided C neuron
            if c_neuron == ffi.NULL:
                raise ValueError("Received null LSTM neuron pointer")
            neuron_ptr = c_neuron
        else:
            # Create new C neuron
            neuron_ptr = lib.lstm_neuron_create(input_size, activation_function)
            if neuron_ptr == ffi.NULL:
                raise RuntimeError("Failed to create LSTM neuron")

        super().__init__(neuron_type, neuron_ptr, activation_function)

    def activate_neuron(self, input_tensor: Tensor) -> float:
        if not isinstance(input_tensor, Tensor):
            raise TypeError("Input must be a Tensor object")

        try:
            result = lib.lstm_neuron_activation(input_tensor.c_tensor, self.neuron)
            return float(result)
        except Exception as e:
            raise RuntimeError(f"LSTM neuron activation failed: {e}")

    def backward_neuron(self):
        try:
            return lib.lstm_neuron_backward()
        except Exception as e:
            raise RuntimeError(f"LSTM neuron backward pass failed: {e}")

    def __del__(self):
        # Conservative approach - let the layer handle cleanup
        pass