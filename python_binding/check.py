from python_binding.tasks import ffi, lib
from neuron import *
from layer import *
from Tensor import Tensor

sigmoid = lib.Sigmoid_function(9)

print("sigmoid answer: \n", sigmoid)

nn = lib.neuron_create(10,2)
print(nn)
np = DenseNeuron(10,c_neuron=nn)
print(np)
print(np.neuron)

n = LstmNeuron(10,lib.LINEAR)
print(n)

input = Tensor(10)
k = n.activate_neuron(input)
print(k)

dn = DenseNeuron(10,lib.LINEAR)
o = dn.activate_neuron(input)
print(o)

Rn = RnnNeuron(10,lib.LINEAR)
L = Rn.activate_neuron(input)
print(L)

rl = RnnLayer(5, 10)
print(rl)
m = rl.layer_forward(input)
print(m)

print(rl.get_neuron(3))