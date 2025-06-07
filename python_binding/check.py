from python_binding.tasks import ffi, lib
from neuron import *
from layer import *
from network import *
from MyTensor import Tensor
import numpy as np


def test_basic_functions():
    """Test basic library functions"""
    print("=== Testing Basic Functions ===")

    try:
        sigmoid = lib.Sigmoid_function(9)
        print(f"Sigmoid(9): {sigmoid}")
    except Exception as e:
        print(f"Error testing sigmoid: {e}")


def test_neurons():
    """Test neuron creation and activation"""
    print("\n=== Testing Neurons ===")

    try:
        # Test Dense Neuron
        print("Testing Dense Neuron...")
        nn = lib.neuron_create(10, 2)
        if nn != ffi.NULL:
            nl = DenseNeuron(10, c_neuron=nn)
            print(f"Dense neuron created: {nl}")
            print(f"Neuron pointer: {nl.neuron}")
        else:
            print("Failed to create dense neuron")
    except Exception as e:
        print(f"Error testing dense neuron: {e}")

    try:
        # Test LSTM Neuron
        print("Testing LSTM Neuron...")
        n = LstmNeuron(10, ActivationType.LINEAR)
        print(f"LSTM neuron created: {n}")
    except Exception as e:
        print(f"Error testing LSTM neuron: {e}")

    try:
        # Test RNN Neuron
        print("Testing RNN Neuron...")
        rn = RnnNeuron(10, ActivationType.LINEAR)
        print(f"RNN neuron created: {rn}")
    except Exception as e:
        print(f"Error testing RNN neuron: {e}")


def test_tensors():
    """Test tensor operations"""
    print("\n=== Testing Tensors ===")

    try:
        # Test basic tensor creation
        print("Creating basic tensor...")
        input_tensor = Tensor.list_to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        print(f"Tensor created: {input_tensor()}")

        # Test tensor printing (if available)
        try:
            lib.tensor_print(input_tensor.c_tensor)
        except Exception as e:
            print(f"Could not print tensor via C function: {e}")

        a = Tensor.list_to_tensor([[1,2],
                                   [3,4]])
        b = Tensor.list_to_tensor([[5,6],
                                   [7,8]])
        aplusb = a + b
        ab = a * b
        aminusb = a-b
        amat = a @ b

        print(f"tensor a {a()}\ntensor b {b()}\ntensor a + b {aplusb()}\ntensor a - b {aminusb()}\ntensor a * b {ab()}\ntensor a @ b {amat()}\n")

        iter_test = Tensor.list_to_tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
        print(f"Tensor created: {iter_test()}")
        for dim in iter_test:
            print(f"dim = {dim()}")
            for mini_dim in dim:
                print(f"    mini_dim = {mini_dim()}")

    except Exception as e:
        print(f"Error creating basic tensor: {e}")

    try:
        # Test numpy conversion
        print("Testing numpy conversion...")
        shape = (2, 3, 4)
        m = np.random.randn(*shape)
        mm = Tensor.from_numpy(m)
        print(f"Numpy tensor: {mm()}")

        # Test slicing
        print("Testing tensor slicing...")
        t_slice = mm[0:1]
        print(f"Slice [0:1]: {t_slice()}")

        t_slice.squeeze()
        print(f"After squeeze: {t_slice()}")

    except Exception as e:
        print(f"Error testing numpy/slicing: {e}")


def test_layers():
    """Test layer operations"""
    print("\n=== Testing Layers ===")

    try:
        # Test Dense Layer
        print("Testing Dense Layer...")
        input_tensor = Tensor.list_to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        dn = DenseLayer(10, 5, ActivationType.LINEAR)
        o = dn.layer_forward(input_tensor)
        print(f"Dense layer output: {o}")

    except Exception as e:
        print(f"Error testing dense layer: {e}")

    try:
        # Test RNN Layer
        print("Testing RNN Layer...")
        input2 = Tensor.list_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        rl = RnnLayer(10, 5)
        print(f"RNN layer created: {rl}")

        m = rl.layer_forward(input2)
        print(f"RNN layer outputs: {len(m)} time steps")

        # Test getting specific neuron
        neuron_3 = rl.get_neuron(3)
        print(f"Neuron 3: {neuron_3}")

    except Exception as e:
        print(f"Error testing RNN layer: {e}")


def test_models():
    """Test complete models"""
    print("\n=== Testing Models ===")

    try:
        # Test simple dense model
        print("Testing dense model...")
        input_tensor = Tensor.list_to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        check = checkModel()
        output = check.forward(input_tensor)
        print(f"Dense model output: {check}, {output}")

    except Exception as e:
        print(f"Error testing dense model: {e}")

    try:
        # Test RNN + Dense model
        print("Testing RNN+Dense model...")
        input2 = Tensor.list_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        check2 = check2Model()
        output = check2.forward(input2)
        print(f"RNN+Dense model output: {check2}, {output}")

        # Test loss calculation and backpropagation
        real = Tensor.list_to_tensor([1])
        loss = NetworkModel.loss_function(LossType.MSE, output, real)
        print(f"Loss: {loss}")

        check2.backward(output, real)
        print("Backward pass completed")

    except Exception as e:
        print(f"Error testing RNN+Dense model: {e}")


def main():
    """Run all tests"""
    print("Starting comprehensive testing...")

    try:
        test_basic_functions()
        test_neurons()
        test_tensors()
        test_layers()
        test_models()

        print("\n=== All Tests Completed ===")

    except Exception as e:
        print(f"Testing failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()