from abc import abstractmethod
from typing import overload, TypeVar, Iterable
_T = TypeVar('_T')

from python_binding.tasks import ffi, lib
import numpy as np
from collections.abc import MutableSequence


class Tensor(MutableSequence):
    def __init__(self, c_tensor=None, flatten: list = None, shape: list = None, dims: int = 2):
        super(Tensor, self).__init__()
        self.c_tensor = c_tensor
        self.flatten = flatten
        self.shape = shape
        self.dims = dims

    @staticmethod
    def c_to_tensor(c_tensor):
        flatten = [c_tensor.data[i] for i in range(c_tensor.count)]
        shape = [c_tensor.shape[i] for i in range(c_tensor.dims)]
        return Tensor(c_tensor, flatten, shape, c_tensor.dims)

    @staticmethod
    def list_to_tensor(values: list):
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
                flatten.extend(vals)
                return [len(vals)]

        shape = [1]
        shape.extend(recurse(values))

        return len(shape), shape, flatten

    def __call__(self):
        print(f"tensor dims: {self.dims} ,shape: {self.shape}")
        lib.tensor_print(self.c_tensor)

    def insert(self, index: int, value: _T) -> None:
        pass

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> _T: ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[_T]: ...

    def __getitem__(self, i: int) -> _T:
        pass

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: _T) -> None: ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[_T]) -> None: ...

    def __setitem__(self, i: int, o: _T) -> None:
        pass

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None: ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None: ...

    def __delitem__(self, i: int) -> None:
        pass

    def __len__(self) -> int:
        pass
