from python_binding.tasks import ffi, lib
from neuron import *
from layer import *
from network import *
from MyTensor import Tensor
import numpy as np

class ExampleModel(NetworkModel):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.lr = 0.01
        self.input_layer = DenseLayer(1, 10, lib.SIGMOID)
        self.hidden_layer = DenseLayer(10, 10, lib.RELU)
        self.output_layer = DenseLayer(10, 1, lib.TANH)
        self.layers.extend([self.input_layer, self.hidden_layer, self.output_layer])

    def forward(self, data: Tensor):
        x = self.input_layer.layer_forward(data)
        h1 = self.hidden_layer.layer_forward(x)
        o = self.output_layer.layer_forward(h1)
        return o

    def backward(self, output: Tensor, real_output: Tensor):
        grad = NetworkModel.loss_derivative(lib.MSE, output, real_output)
        super().backward(grad)

def main():
        try:
            # Generate synthetic regression data
            print("Generating synthetic data...")
            X = np.random.rand(1000, 1) * 10  # 1000 samples, 1 feature
            y = 2 * X[:, 0] + 3 + np.random.randn(1000) * 0.5  # y = 2x + 3 + noise

            # Scale X to [0, 1]
            X_min, X_max = X.min(), X.max()
            X = (X - X_min) / (X_max - X_min)

            # Scale y to [0, 1]
            y_min, y_max = y.min(), y.max()
            y = (y - y_min) / (y_max - y_min)

            # Convert to Tensors
            print("Converting data to tensors...")
            data_tensor = Tensor.from_numpy(X)
            real_tensor = Tensor.from_numpy(y.reshape(-1, 1))

            print(f"Data tensor shape: {data_tensor.shape}")
            print(f"Target tensor shape: {real_tensor.shape}")

            # Instantiate model
            print("Creating model...")
            model = ExampleModel()

            # Training loop
            epochs = 1000  # Reduced for testing
            print(f"Starting training for {epochs} epochs...")

            for epoch in range(epochs):
                try:
                    # Get a single sample (ensure safe indexing)
                    sample_idx = epoch % len(data_tensor.flatten)
                    sample = data_tensor[sample_idx] if data_tensor.dims > 1 else Tensor.list_to_tensor(
                        [data_tensor.flatten[sample_idx]])
                    label = real_tensor[sample_idx] if real_tensor.dims > 1 else Tensor.list_to_tensor(
                        [real_tensor.flatten[sample_idx]])

                    # Ensure samples are proper tensors
                    if not isinstance(sample, Tensor):
                        sample = Tensor.list_to_tensor([sample])
                    if not isinstance(label, Tensor):
                        label = Tensor.list_to_tensor([label])

                    # Forward pass
                    output = model.forward(sample)

                    # Calculate loss
                    loss = NetworkModel.loss_function(lib.MSE, output, label)

                    # Print progress
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}, Loss: {loss:.6f}")

                    # Backward pass
                    model.backward(output, label)

                except Exception as e:
                    print(f"Error at epoch {epoch}: {e}")
                    break

            print("Training completed!")

            # Test the model with a few predictions
            print("\nTesting model predictions:")
            test_inputs = [1.0, 5.0, 8.0]
            for test_val in test_inputs:
                try:
                    # Normalize input to match training scaling
                    norm_test_val = (test_val - X_min) / (X_max - X_min)
                    test_tensor = Tensor.list_to_tensor([norm_test_val])
                    prediction = model.forward(test_tensor)
                    # De-normalize prediction for comparison
                    pred_val = prediction.flatten[0] * (y_max - y_min) + y_min
                    expected = 2 * test_val + 3
                    print(f"Input: {test_val}, Predicted: {pred_val:.3f}, Expected: {expected:.3f}")
                except Exception as e:
                    print(f"Error testing with input {test_val}: {e}")

        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

