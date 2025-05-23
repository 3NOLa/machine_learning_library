from python_binding.tasks import ffi, lib
from neuron import *
from layer import *
from MyTensor import Tensor
from typing import *


class NetworkModel:
    def __init__(self):
        self.cuda = False
        self.num_layers = 0
        self.lr = 0.01
        self.layers: List[Layer] = []

    def forward(self, data: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward method")

    def backward(self, output_gradients: Tensor):
        """Backward pass through all layers in reverse order"""
        if not isinstance(output_gradients, Tensor):
            raise TypeError("Output gradients must be a Tensor object")

        grads = output_gradients
        for layer in reversed(self.layers):
            try:
                grads = layer.layer_backward(grads, self.lr)
            except Exception as e:
                print(f"Error in backward pass for layer {type(layer).__name__}: {e}")
                raise

    def update_weights(self):
        for layer in self.layers:
            layer.update_layer_weights(self.lr)

    def network_grad_reset(self):
        for layer in self.layers:
            layer.layer_grad_zero()

    @staticmethod
    def loss_function(loss_type, y_pred: Tensor, y_real: Tensor) -> float:
        """Calculate loss between predicted and real values"""
        if not isinstance(y_pred, Tensor) or not isinstance(y_real, Tensor):
            raise TypeError("Both y_pred and y_real must be Tensor objects")

        try:
            loss_value = lib.loss_active_function(loss_type, y_pred.c_tensor, y_real.c_tensor)
            return float(loss_value)
        except Exception as e:
            raise RuntimeError(f"Loss function calculation failed: {e}")

    @staticmethod
    def loss_derivative(loss_type, y_pred: Tensor, y_real: Tensor) -> Tensor:
        """Calculate loss derivative for backpropagation"""
        if not isinstance(y_pred, Tensor) or not isinstance(y_real, Tensor):
            raise TypeError("Both y_pred and y_real must be Tensor objects")

        try:
            c_result = lib.loss_derivative_active_function(loss_type, y_pred.c_tensor, y_real.c_tensor)
            if c_result == ffi.NULL:
                raise RuntimeError("Loss derivative calculation returned null")
            return Tensor.from_c_tensor(c_result)
        except Exception as e:
            raise RuntimeError(f"Loss derivative calculation failed: {e}")


class Network(NetworkModel):
    def __init__(self, layers: List[Layer]):
        super().__init__()

        if not layers:
            raise ValueError("Network must have at least one layer")

        # Create C network
        self.c_network = lib.network_create_empty()
        if self.c_network == ffi.NULL:
            raise RuntimeError("Failed to create C network")

        self.c_network.input_dims = layers[0].input_dim
        self.set_train(layers[0].type)

        # Add layers to the network
        for layer in layers:
            try:
                lib.add_created_layer(self.c_network, layer.layer_ptr)
                self.layers.append(layer)
                self.num_layers += 1
            except Exception as e:
                raise RuntimeError(f"Failed to add layer {type(layer).__name__}: {e}")

        # Set default loss function
        self.loss = {
            "type": lib.MSE,
            "forward": lib.squared_error_net,
            "backward": lib.derivative_squared_error_net
        }
        self.set_loss(lib.MSE)

    def set_loss(self, loss_type):
        """Set the loss function for the network"""
        try:
            self.c_network.lossFunction = loss_type
            lib.set_loss_function(self.c_network, loss_type)

            self.loss["type"] = loss_type
            self.loss["forward"] = self.c_network.LossFuntionPointer
            self.loss["backward"] = self.c_network.LossDerivativePointer
        except Exception as e:
            raise RuntimeError(f"Failed to set loss function: {e}")

    def set_train(self, train_type):
        """Set the training type for the network"""
        try:
            self.c_network.type = train_type
            lib.network_train_type(self.c_network)
        except Exception as e:
            raise RuntimeError(f"Failed to set training type: {e}")

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass through the entire network"""
        if not isinstance(data, Tensor):
            raise TypeError("Input data must be a Tensor object")

        try:
            c_result = lib.forwardPropagation(self.c_network, data.c_tensor)
            if c_result == ffi.NULL:
                raise RuntimeError("Forward propagation returned null")
            return Tensor.from_c_tensor(c_result)
        except Exception as e:
            raise RuntimeError(f"Forward propagation failed: {e}")

    def backward(self, predictions: Tensor, targets: Tensor):
        """Backward pass through the entire network - Fixed to match C implementation"""
        if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("Both predictions and targets must be Tensor objects")

        try:
            # Use the C library's backpropagation function directly
            # This matches the C implementation which calculates loss derivatives internally
            result = lib.backpropagation(self.c_network, predictions.c_tensor, targets.c_tensor)
            if result == 0:  # C function returns 0 on failure, 1 on success
                raise RuntimeError("C backpropagation function failed")
        except Exception as e:
            raise RuntimeError(f"Backpropagation failed: {e}")

    def train(self, input_data: Tensor, target: Tensor) -> float:
        """Train the network for one iteration - matches C train function"""
        if not isinstance(input_data, Tensor) or not isinstance(target, Tensor):
            raise TypeError("Both input_data and target must be Tensor objects")

        try:
            # Use the C library's train function which handles forward, loss calculation, and backward
            error = lib.train(self.c_network, input_data.c_tensor, target.c_tensor)
            if error < 0:  # C function returns negative on failure
                raise RuntimeError("Training iteration failed")
            return float(error)
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

    def __del__(self):
        """Cleanup network resources"""
        if hasattr(self, 'c_network') and self.c_network and self.c_network != ffi.NULL:
            try:
                lib.network_free(self.c_network)
            except:
                pass
            self.c_network = None


class checkModel(Network):
    """Example model for testing dense layers"""

    def __init__(self):
        self.layer1 = DenseLayer(10, 10, lib.RELU)
        self.layer2 = DenseLayer(10, 5)
        self.outputlayer = DenseLayer(5, 1, lib.SIGMOID)

        super(checkModel, self).__init__([self.layer1, self.layer2, self.outputlayer])


class check2Model(NetworkModel):
    """Example model combining RNN and Dense layers - Fixed backward implementation"""

    def __init__(self):
        super(check2Model, self).__init__()
        self.input_layer = RnnLayer(10, 10, lib.SIGMOID)
        self.hidden_layer = DenseLayer(10, 3)
        self.output_layer = DenseLayer(3, 1, lib.TANH)

        self.layers.extend([self.input_layer, self.hidden_layer, self.output_layer])

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass through RNN + Dense layers"""
        if not isinstance(data, Tensor):
            raise TypeError("Input data must be a Tensor object")

        try:
            # RNN layer returns list of outputs for each time step
            x = self.input_layer.layer_forward(data)
            print(f"RNN outputs: {len(x)} time steps")

            # Use the last time step output
            if not x:
                raise RuntimeError("RNN layer produced no outputs")

            last_output = x[-1]  # Take the last time step
            h1 = self.hidden_layer.layer_forward(last_output)
            o = self.output_layer.layer_forward(h1)
            print(f"Final output: {o}")

            return o
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {e}")

    def backward(self, output: Tensor, real_output: Tensor):
        """Backward pass with gradient calculation - Fixed to properly propagate gradients"""
        if not isinstance(output, Tensor) or not isinstance(real_output, Tensor):
            raise TypeError("Both output and real_output must be Tensor objects")

        try:
            # Calculate the gradient of the loss with respect to the output
            grad = NetworkModel.loss_derivative(lib.MSE, output, real_output)
            print(f"Initial gradient: {grad}")

            self.lr = 0.1

            # Properly propagate gradients through each layer in reverse order
            current_grad = grad

            # Backward through output layer
            current_grad = self.output_layer.layer_backward(current_grad, self.lr)

            # Backward through hidden layer
            current_grad = self.hidden_layer.layer_backward(current_grad, self.lr)

            # For RNN layer, we need to handle the sequence differently
            # The RNN backward should handle the temporal dependencies
            self.input_layer.layer_backward(current_grad, self.lr)

        except Exception as e:
            raise RuntimeError(f"Backward pass failed: {e}")


# Additional utility class for RNN training
class RNNNetwork(NetworkModel):
    """Network specifically designed for RNN training with proper sequence handling"""

    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers
        self.num_layers = len(layers)

    def forward(self, sequence_data: Tensor) -> Tensor:
        """Forward pass through sequence data"""
        if not isinstance(sequence_data, Tensor):
            raise TypeError("Input data must be a Tensor object")

        current_output = sequence_data

        for i, layer in enumerate(self.layers):
            if isinstance(layer, (RnnLayer, LstmLayer)):
                # RNN/LSTM layers return list of outputs
                outputs = layer.layer_forward(current_output)
                # Use last output for next layer
                current_output = outputs[-1] if outputs else None
                if current_output is None:
                    raise RuntimeError(f"Layer {i} produced no output")
            else:
                # Dense layers return single tensor
                current_output = layer.layer_forward(current_output)

        return current_output

    def backward(self, predictions: Tensor, targets: Tensor):
        """Backward pass for sequence models"""
        if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("Both predictions and targets must be Tensor objects")

        # Calculate initial gradient
        grad = NetworkModel.loss_derivative(lib.MSE, predictions, targets)

        # Propagate backwards through layers
        current_grad = grad
        for layer in reversed(self.layers):
            current_grad = layer.layer_backward(current_grad, self.lr)

    def train_sequence(self, sequence_input: Tensor, target: Tensor) -> float:
        """Train on a sequence with proper error calculation"""
        # Forward pass
        predictions = self.forward(sequence_input)

        # Calculate loss
        loss = NetworkModel.loss_function(lib.MSE, predictions, target)

        # Backward pass
        self.backward(predictions, target)

        return loss