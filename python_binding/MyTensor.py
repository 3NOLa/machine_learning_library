from abc import abstractmethod
from typing import overload, TypeVar, Iterable
from python_binding.tasks import ffi, lib
import numpy as np
import pandas as pd
from collections.abc import MutableSequence

_T = TypeVar('_T')

class Tensor(MutableSequence):
    def __init__(self, c_tensor=None, flatten: list[float] = None, shape: list = None, dims: int = 1):
        super(Tensor, self).__init__()
        self.c_tensor = c_tensor
        self.flatten = flatten
        self.shape = tuple(shape)
        self.dims = dims

    @staticmethod
    def from_c_tensor(c_tensor) -> 'Tensor':
        flatten = [c_tensor.data[i] for i in range(c_tensor.count)]
        shape = [c_tensor.shape[i] for i in range(c_tensor.dims)]
        return Tensor(c_tensor, flatten, shape, c_tensor.dims)

    @staticmethod
    def list_to_tensor(values: list) -> 'Tensor':
        dims, shape, flatten = Tensor.get_tensor_info(values)

        c_flatten = ffi.new("float []", flatten)
        c_shape = ffi.new("int []", shape)

        c_tensor = lib.tensor_create_flatten(dims, c_shape, c_flatten, len(flatten))

        return Tensor(c_tensor, flatten, shape, dims)

    @staticmethod
    def get_tensor_info(values):
        flatten = []

        def recurse(vals): # dfs for finding tesnor info
            if isinstance(vals[0], list):
                sub_shape = recurse(vals[0])
                for sub in vals:
                    if recurse(sub) != sub_shape:
                        raise ValueError("Ragged tensor detected")
                return [len(vals)] + sub_shape
            else:
                for x in vals:
                    if not isinstance(x, (int, float)):
                        raise TypeError(f"Non-numeric tensor element: {x!r}")
                    flatten.append(float(x))
                return [len(vals)]

        shape = recurse(values)

        return len(shape), shape, flatten

    @staticmethod
    def from_numpy(np_array : np) -> 'Tensor':
        if not isinstance(np_array, np.ndarray):
            raise TypeError("input must be a NumPy ndarray")

        flatten = np_array.flatten().astype(np.float32).tolist()
        shape = list(np_array.shape)
        dims = len(shape)

        c_flatten = ffi.new("float []", flatten)
        c_shape = ffi.new("int []", shape)

        c_tensor = lib.tensor_create_flatten(dims, c_shape, c_flatten, len(flatten))
        return Tensor(c_tensor, flatten, shape, dims)

    @staticmethod
    def from_pandas(df : pd) -> 'Tensor':
        return Tensor.from_numpy(df.to_numpy())

    def squeeze(self):
        lib.tensor_squeeze(self.c_tensor)
        self.dims = self.c_tensor.dims
        self.shape = tuple([self.c_tensor.shape[i] for i in range(self.dims)])

    def __len__(self) -> int:
        return len(self.flatten)

    def getslice(self, obj_slice: slice) -> 'Tensor':
        return Tensor.from_c_tensor(lib.tensor_slice_range(self.c_tensor,obj_slice.start,obj_slice.stop))

    def getrow(self, index: int) -> 'Tensor':
        return Tensor.from_c_tensor(lib.tensor_get_row(self.c_tensor,index))

    def __getitem__(self, index: int) -> _T:
        if isinstance(index, tuple):
            return self.getslice()
        elif isinstance(index, slice):
            return self.getslice(index)
        else:
            return self.getrow(index) if self.dims > 1 else self.flatten[index]

    def __call__(self):
        return f"Tensor: ({self.flatten}, tensor dims: {self.dims} ,shape: {self.shape})"


    def __del__(self):
        if self.c_tensor:
            lib.tensor_free(self.c_tensor)

    def __setitem__(self, s: slice, o: Iterable[_T]) -> None:
        pass

    def insert(self, index: int, value: _T) -> None:
        pass

    def __delitem__(self, i: int) -> None:
        pass

