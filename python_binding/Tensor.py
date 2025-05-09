from python_binding.tasks import ffi, lib
import numpy as np


class Tensor:
    def __init__(self,size : int, data=None):
        if isinstance(data, list) or isinstance(data, np.ndarray):
            self.CTensor = lib.tensor_create(data)
        else:
            shape = ffi.new("int []",[1,size])
            self.CTensor = lib.tensor_create(2,shape)

    def CTensor(self):
        return self.CTensor

