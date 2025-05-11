from python_binding.tasks import ffi, lib
from neuron import *
from layer import *
from Tensor import Tensor
from typing import *


class NetworkModel:
    def __init__(self):
        self.cuda = False
        self.num_layers = 0
        self.lr = 0.01
        self.layers = []

    def forward(self, data:Tensor):
        raise NotImplementedError

    def backward(self, output_gradients:Tensor):
        grads = output_gradients
        for layer in reversed(self.layers):
            grads = layer.layer_backward(grads,self.lr)

    @staticmethod
    def loss_function(loss, y_pred: Tensor, y_real:Tensor):
        return lib.loss_active_function(loss,y_pred.CTensor,y_real.CTensor)

    @staticmethod
    def loss_derivative(loss, y_pred: Tensor, y_real:Tensor):
        return Tensor(1,lib.loss_derivative_active_function(loss,y_pred.CTensor,y_real.CTensor))


class Network(NetworkModel):
    def __init__(self, layers : [Layer]):
        super().__init__()
        self.c_network = lib.network_create_empty()
        self.c_network.input_dims = layers[0].input_dim

        self.set_train(layers[0].type)

        for layer in layers:
            lib.add_created_layer(self.c_network,layer.layer_ptr)
            self.layers.append(layer)
            self.num_layers += 1

        self.loss = {
            "type": lib.MSE,
            "forward": lib.squared_error_net,
            "backward": lib.derivative_squared_error_net
        }

    def set_loss(self, loss_type):
        self.c_network.lossFunction = loss_type
        lib.set_loss_function(self.c_network, loss_type)

        self.loss["type"] = loss_type
        self.loss["forward"] = self.c_network.LossFuntionPointer
        self.loss["backward"] = self.c_network.LossDerivativePointer

    def set_train(self, train_type):
        self.c_network.type = train_type
        lib.network_train_type(self.c_network)

    def forward(self, data: Tensor):
        return lib.forwardPropagation(self.c_network, data.CTensor)

    def backward(self, predictions:Tensor, targets:Tensor):
        return lib.backpropagation(self.c_network, predictions.CTensor, targets.CTensor)


class checkModel(Network):
    def __init__(self):
        self.layer1 = DenseLayer(10,10,lib.RELU)
        self.layer2 = DenseLayer(5,10)
        self.outputlayer = DenseLayer(1,5,lib.SIGMOID)

        super(checkModel, self).__init__([self.layer1,self.layer2,self.outputlayer])


class check2Model(NetworkModel):
    def __init__(self):
        super(check2Model, self).__init__()
        self.input_layer = RnnLayer(10,10,lib.SIGMOID)
        self.hidden_layer = DenseLayer(3,10)
        self.output_layer = DenseLayer(1,3,lib.TANH)

        self.layers.extend([self.input_layer,self.hidden_layer,self.output_layer])

    def forward(self, data:Tensor):
        x = self.input_layer.layer_forward(data,5)
        print(x)
        h1 = self.hidden_layer.layer_forward(x[-1])
        o = self.output_layer.layer_forward(h1)

        return o

    def backward(self, ouptut:Tensor,real_output:Tensor):
        grad = NetworkModel.loss_derivative(lib.MSE, ouptut, real_output)
        print("grad" , grad)
        self.lr = 0.1
        super().backward(grad)
