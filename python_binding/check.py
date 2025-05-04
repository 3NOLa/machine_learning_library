from python_binding.tasks import ffi, lib

# Create target array

sigmoid = lib.Sigmoid_function(9)

print("sigmoid answer: \n", sigmoid)

neuron = lib.neuron_create(10,2)
print(neuron)