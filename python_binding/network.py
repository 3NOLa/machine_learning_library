from python_binding.tasks import ffi, lib
from neuron import *
from layer import *
import Tensor
from typing import *


class NetworkModel:
    def __init__(self):
        self.cuda = False
        self.num_layers = 0
        self.layers = [Layer]
        self.loss = {
            "type": lib.MSE,
            "forward": lib.squared_error_net,
            "backward": lib.derivative_squared_error_net
        }

    def forward(self, data:Tensor):
        raise NotImplementedError

    def backward(self, predictions:Tensor, targets:Tensor):
        raise NotImplementedError


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
        super.__init__()
        self.input_layer = RnnLayer(10,10,lib.SIGMOID)
        self.hidden_layer = DenseLayer()