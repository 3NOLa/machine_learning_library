from python_binding.cbinding.tasks import ffi
from .neuron import *
from .MyTensor import Tensor
from .py_enums import *
from typing import List


class Layer:
    def __init__(self, layer_type: LayerType, input_dim: int, neuron_amount: int, activation_type: ActivationType = None, initializer_type: InitializerType = None):
        self.layer_type_ptr = None
        self.type = layer_type
        self.input_dim = input_dim
        self.activation_function = ActivationType.LINEAR if activation_type is None else activation_type
        self.initializer_type = InitializerType.XavierNormal if initializer_type is None else initializer_type
        self.neuron_amount = neuron_amount

        # Initialize the general layer
        self.layer_ptr = lib.general_layer_Initialize(
            self.type,
            self.neuron_amount,
            self.input_dim,
            self.activation_function
        )

        if self.layer_ptr == ffi.NULL:
            raise RuntimeError("Failed to initialize layer")

        self.py_neurons = []

    def layer_forward(self, input_tensor: Tensor) -> Tensor:
        if not isinstance(input_tensor, Tensor):
            raise TypeError("Input must be a Tensor object")

        c_result = lib.layer_ptr.forward(self.layer_ptr, input_tensor.c_tensor)
        if c_result == ffi.NULL:
            raise RuntimeError("Layer forward pass failed")
        return Tensor.from_c_tensor(c_result)

    def layer_backward(self, input_gradients: Tensor, lr: float) -> Tensor:
        if not isinstance(input_gradients, Tensor):
            raise TypeError("Input gradients must be a Tensor object")

        c_result = lib.layer_ptr.backward(self.layer_ptr, input_gradients.c_tensor, lr)
        if c_result == ffi.NULL:
            raise RuntimeError("Layer backward pass failed")
        return Tensor.from_c_tensor(c_result)

    def get_neuron(self, index: int):
        if index < 0 or index >= len(self.py_neurons):
            raise IndexError(f"Neuron index {index} out of bounds")
        return self.py_neurons[index]

    def layer_grad_zero(self):
        raise TypeError("Subclasses must implement forward method")

    def update_layer_weights(self, lr: float):
        raise TypeError("Subclasses must implement forward method")

    def set_layer_optimizer(self, optimizer_type: OptimizerType):
        lib.set_layer_optimizer(self.layer_ptr, optimizer_type)

    def set_layer_initializer(self, initializer_type: InitializerType):
        initializer = ffi.cast("Initializer *", ffi.NULL)
        self.layer_ptr.opt_init(self.layer_ptr,initializer , initializer_type)

    def __del__(self):
        if hasattr(self, 'layer_ptr') and self.layer_ptr and self.layer_ptr != ffi.NULL:
            try:
                lib.general_layer_free(self.layer_ptr)
            except:
                pass
            self.layer_ptr = None


class DenseLayer(Layer):
    def __init__(self, input_dim: int, neuron_amount: int, activation_type: ActivationType = None, initializer_type: InitializerType = None):
        super().__init__(LayerType.LAYER_DENSE, input_dim, neuron_amount, activation_type, initializer_type)

        self.layer_type_ptr = lib.layer_create(neuron_amount, input_dim, self.activation_function)
        if self.layer_type_ptr == ffi.NULL:
            raise RuntimeError("Failed to create dense layer")

        self.set_layer_initializer(self.initializer_type) #must call it after c layer init

        self.py_neurons = []
        for i in range(neuron_amount):
            try:
                neuron_ptr = self.layer_type_ptr.neurons[i]
                if neuron_ptr != ffi.NULL:
                    self.py_neurons.append(
                        DenseNeuron(
                            input_size=input_dim,
                            c_neuron=neuron_ptr,
                            activation_type=activation_type
                        )
                    )
            except Exception as e:
                print(f"Warning: Could not create neuron {i}: {e}")

    def layer_forward(self, input_tensor: Tensor) -> Tensor:
        if not isinstance(input_tensor, Tensor):
            raise TypeError("Input must be a Tensor object")

        c_result = lib.layer_forward(self.layer_type_ptr, input_tensor.c_tensor)
        if c_result == ffi.NULL:
            raise RuntimeError("Dense layer forward pass failed")
        return Tensor.from_c_tensor(c_result)

    def layer_backward(self, input_gradients: Tensor, lr: float) -> Tensor:
        if not isinstance(input_gradients, Tensor):
            raise TypeError("Input gradients must be a Tensor object")

        c_result = lib.layer_backward(self.layer_type_ptr, input_gradients.c_tensor)
        if c_result == ffi.NULL:
            raise RuntimeError("Dense layer backward pass failed")
        return Tensor.from_c_tensor(c_result)

    def update_layer_weights(self, lr: float):
        lib.dense_layer_update(self.layer_type_ptr, lr)

    def layer_grad_zero(self):
        lib.dense_layer_zero_grad(self.layer_type_ptr)


class RnnLayer(Layer):
    def __init__(self, input_dim: int, neuron_amount: int, activation_type: ActivationType = None, initializer_type: InitializerType = None):
        activation_type = activation_type if activation_type is not None else lib.LINEAR
        super().__init__(LayerType.LAYER_RNN, input_dim, neuron_amount, activation_type, initializer_type)

        # Create the specific RNN layer
        self.layer_type_ptr = lib.rnn_layer_create(neuron_amount, input_dim, self.activation_function)
        if self.layer_type_ptr == ffi.NULL:
            raise RuntimeError("Failed to create RNN layer")

        self.set_layer_initializer(self.initializer_type) #must call it after c layer init


        # Create Python neuron wrappers
        self.py_neurons = []
        for i in range(neuron_amount):
            try:
                neuron_ptr = self.layer_type_ptr.neurons[i]
                if neuron_ptr != ffi.NULL:
                    self.py_neurons.append(
                        RnnNeuron(
                            input_size=input_dim,
                            c_neuron=neuron_ptr,
                            activation_type=activation_type
                        )
                    )
            except Exception as e:
                print(f"Warning: Could not create RNN neuron {i}: {e}")

    def layer_forward(self, input_tensor: Tensor) -> List[Tensor]:
        if not isinstance(input_tensor, Tensor):
            raise TypeError("Input must be a Tensor object")

        if input_tensor.dims < 2:
            raise ValueError("RNN layer requires at least 2D input (time_steps, features)")

        t_outputs = []
        time_steps = input_tensor.shape[0]

        for t in range(time_steps):
            try:
                input_t = input_tensor[t]  # Get slice for time step t
                if isinstance(input_t, Tensor):
                    input_t.squeeze()  # Remove unnecessary dimensions
                    c_result = lib.rnn_layer_forward(self.layer_type_ptr, input_t.c_tensor)
                    if c_result != ffi.NULL:
                        t_outputs.append(Tensor.from_c_tensor(c_result))
                    else:
                        raise RuntimeError(f"RNN forward pass failed at time step {t}")
                else:
                    raise TypeError(f"Expected Tensor slice at time step {t}")
            except Exception as e:
                print(f"Error processing time step {t}: {e}")
                raise

        return t_outputs

    def layer_backward(self, input_gradients: Tensor, lr: float) -> Tensor:
        if not isinstance(input_gradients, Tensor):
            raise TypeError("Input gradients must be a Tensor object")

        c_result = lib.rnn_layer_backward(self.layer_type_ptr, input_gradients.c_tensor)
        if c_result == ffi.NULL:
            raise RuntimeError("RNN layer backward pass failed")
        return Tensor.from_c_tensor(c_result)

    def update_layer_weights(self, lr: float):
        lib.rnn_layer_update(self.layer_type_ptr, lr)

    def layer_grad_zero(self):
        lib.rnn_layer_zero_grad(self.layer_type_ptr)


class LstmLayer(Layer):
    def __init__(self, input_dim: int, neuron_amount: int, activation_type: ActivationType = None, initializer_type: InitializerType = None):
        super().__init__(LayerType.LAYER_LSTM, input_dim, neuron_amount, activation_type, initializer_type)

        # Create the specific LSTM layer
        self.layer_type_ptr = lib.lstm_layer_create(neuron_amount, input_dim, self.activation_function)
        if self.layer_type_ptr == ffi.NULL:
            raise RuntimeError("Failed to create LSTM layer")

        self.set_layer_initializer(self.initializer_type) #must call it after c layer init

        # Create Python neuron wrappers
        self.py_neurons = []
        for i in range(neuron_amount):
            try:
                neuron_ptr = self.layer_type_ptr.neurons[i]
                if neuron_ptr != ffi.NULL:
                    self.py_neurons.append(
                        LstmNeuron(
                            input_size=input_dim,
                            c_neuron=neuron_ptr,
                            activation_type=activation_type
                        )
                    )
            except Exception as e:
                print(f"Warning: Could not create LSTM neuron {i}: {e}")

    def layer_forward(self, input_tensor: Tensor) -> List[Tensor]:
        if not isinstance(input_tensor, Tensor):
            raise TypeError("Input must be a Tensor object")

        if input_tensor.dims < 2:
            raise ValueError("LSTM layer requires at least 2D input (time_steps, features)")

        t_outputs = []
        time_steps = input_tensor.shape[0]

        for t in range(time_steps):
            try:
                input_t = input_tensor[t]  # Get slice for time step t
                if isinstance(input_t, Tensor):
                    input_t.squeeze()  # Remove unnecessary dimensions
                    # Note: Using input_tensor.c_tensor here might be incorrect
                    # It should probably be input_t.c_tensor for the specific time step
                    c_result = lib.lstm_layer_forward(self.layer_type_ptr, input_t.c_tensor)
                    if c_result != ffi.NULL:
                        t_outputs.append(Tensor.from_c_tensor(c_result))
                    else:
                        raise RuntimeError(f"LSTM forward pass failed at time step {t}")
                else:
                    raise TypeError(f"Expected Tensor slice at time step {t}")
            except Exception as e:
                print(f"Error processing time step {t}: {e}")
                raise

        return t_outputs

    def layer_backward(self, input_gradients: Tensor, lr: float) -> Tensor:
        if not isinstance(input_gradients, Tensor):
            raise TypeError("Input gradients must be a Tensor object")

        c_result = lib.lstm_layer_backward(self.layer_type_ptr, input_gradients.c_tensor)
        if c_result == ffi.NULL:
            raise RuntimeError("LSTM layer backward pass failed")
        return Tensor.from_c_tensor(c_result)

    def update_layer_weights(self, lr: float):
        lib.lstm_layer_update(self.layer_type_ptr, lr)

    def layer_grad_zero(self):
        lib.lstm_layer_zero_grad(self.layer_type_ptr)