from python_binding.tasks import ffi, lib
from neuron import *
from layer import *
from network import *
from MyTensor import Tensor
import numpy as np

sigmoid = lib.Sigmoid_function(9)

print("sigmoid answer: \n", sigmoid)

nn = lib.neuron_create(10,2)
print(nn)
nl = DenseNeuron(10,c_neuron=nn)
print(nl)
print(nl.neuron)
print("---------------------------")

n = LstmNeuron(10,lib.LINEAR)
print(n)
print("---------------------------")

input = Tensor.list_to_tensor([1,2,3,4,5,6,7,8,9,10])
print(input())
lib.tensor_print(input.c_tensor)
k = n.activate_neuron(input)
print(k)
print("---------------------------")

dn = DenseNeuron(10,lib.LINEAR)
o = dn.activate_neuron(input)
print(o)
print("---------------------------")
check = checkModel()
output = check.forward(input)
print(check,output)

print("---------------------------")
Rn = RnnNeuron(10,lib.LINEAR)
L = Rn.activate_neuron(input)
print(L)
print("---------------------------")

input2 = Tensor.list_to_tensor([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
rl = RnnLayer(10, 5)
print(rl)
m = rl.layer_forward(input2)
print(m)

print(rl.get_neuron(3))

print("---------------------------")

check2 = check2Model()
output = check2.forward(input2)
print(check2,output)
real = Tensor.list_to_tensor([1])
print("erorr: " + str(NetworkModel.loss_function(lib.MSE,output,real)))
check2.backward(output,real)
print("end")
print("---------------------------")

shape = (2,3,4)
m = np.random.randn(*shape)
mm = Tensor.from_numpy(m)
print(mm())
t_slice : Tensor = mm[0:1]
print(t_slice())
t_slice.squeeze()
print(t_slice())

print("---------------------------")
