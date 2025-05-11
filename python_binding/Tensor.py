from python_binding.tasks import ffi, lib
import numpy as np


class Tensor:
    def __init__(self,size : int, data=None):
        self.size = size
        if isinstance(data, list) or isinstance(data, np.ndarray):
            self.CTensor = lib.tensor_create(data)
        elif data is not None and ffi.typeof(data).cname == "Tensor *" :
            self.CTensor = data
        else:
            shape = ffi.new("int []",[1,size])
            self.CTensor = lib.tensor_random_create(2,shape)

    def CTensor(self):
        return self.CTensor

    def __call__(self):
        lib.tensor_print(self.CTensor)