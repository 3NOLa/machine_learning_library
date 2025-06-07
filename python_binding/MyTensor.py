from abc import abstractmethod
from typing import overload, TypeVar, Union, List, Iterable
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
        if c_tensor == ffi.NULL:
            raise ValueError("Received null C tensor")

        flatten = [c_tensor.data[i] for i in range(c_tensor.count)]
        shape = [c_tensor.shape[i] for i in range(c_tensor.dims)]
        return Tensor(c_tensor, flatten, shape, c_tensor.dims)

    @staticmethod
    def list_to_tensor(values: Union[List, List[List]]) -> 'Tensor':
        if not values:
            raise ValueError("Cannot create tensor from empty list")

        dims, shape, flatten = Tensor.get_tensor_info(values)

        c_flatten = ffi.new("float []", flatten)
        c_shape = ffi.new("int []", shape)

        c_tensor = lib.tensor_create_flatten(dims, c_shape, c_flatten, len(flatten))

        if c_tensor == ffi.NULL:
            raise RuntimeError("Failed to create C tensor")

        return Tensor(c_tensor, flatten, shape, dims)

    @staticmethod
    def get_tensor_info(values):
        flatten = []

        def recurse(vals):
            if not vals:
                return []

            if isinstance(vals[0], (list, tuple)):
                sub_shapes = []
                for sub in vals:
                    shape = recurse(sub)
                    sub_shapes.append(shape)
                first_shape = sub_shapes[0]
                for shape in sub_shapes:
                    if shape != first_shape:
                        raise ValueError("Ragged tensor detected")
                return [len(vals)] + first_shape
            else:
                for x in vals:
                    if not isinstance(x, (int, float)):
                        raise TypeError(f"Non-numeric tensor element: {x!r}")
                    flatten.append(float(x))
                return [len(vals)]

        shape = recurse(values)
        return len(shape), shape, flatten

    @staticmethod
    def from_numpy(np_array: np.ndarray) -> 'Tensor':
        if not isinstance(np_array, np.ndarray):
            raise TypeError("input must be a NumPy ndarray")

        flatten = np_array.flatten().astype(np.float32).tolist()
        shape = list(np_array.shape)
        dims = len(shape)

        c_flatten = ffi.new("float []", flatten)
        c_shape = ffi.new("int []", shape)

        c_tensor = lib.tensor_create_flatten(dims, c_shape, c_flatten, len(flatten))

        if c_tensor == ffi.NULL:
            raise RuntimeError("Failed to create C tensor from numpy array")

        return Tensor(c_tensor, flatten, shape, dims)

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> 'Tensor':
        return Tensor.from_numpy(df.to_numpy())

    def squeeze(self):
        if self.c_tensor:
            lib.tensor_squeeze(self.c_tensor)
            self.dims = self.c_tensor.dims
            self.shape = tuple([self.c_tensor.shape[i] for i in range(self.dims)])
            # Update flatten as well
            self.flatten = [self.c_tensor.data[i] for i in range(self.c_tensor.count)]

    def __len__(self) -> int:
        return len(self.flatten)

    def getslice(self, obj_slice: slice) -> 'Tensor':
        start = obj_slice.start if obj_slice.start is not None else 0
        stop = obj_slice.stop if obj_slice.stop is not None else self.shape[0]

        c_result = lib.tensor_slice_range(self.c_tensor, start, stop)
        if c_result == ffi.NULL:
            raise RuntimeError("Failed to slice tensor")
        return Tensor.from_c_tensor(c_result)

    def getrow(self, index: int) -> 'Tensor':
        if index < 0 or (self.dims > 0 and index >= self.shape[0]):
            raise IndexError(f"Index {index} out of bounds for tensor with shape {self.shape}")

        c_result = lib.tensor_get_row(self.c_tensor, index)
        if c_result == ffi.NULL:
            raise RuntimeError(f"Failed to get row {index}")
        return Tensor.from_c_tensor(c_result)

    def __getitem__(self, index: Union[int, slice, tuple]) -> Union['Tensor', float]:
        if isinstance(index, tuple):
            # For multi-dimensional indexing, implement as needed
            raise NotImplementedError("Multi-dimensional indexing not yet implemented")
        elif isinstance(index, slice):
            return self.getslice(index)
        else:
            if self.dims > 1:
                return self.getrow(index)
            else:
                if index < 0 or index >= len(self.flatten):
                    raise IndexError(f"Index {index} out of bounds")
                return self.flatten[index]

    def __iter__(self):
        return self.TensorIterator(self)

    class TensorIterator:
        def __init__(self, outer):
            self.tensor = outer
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index < self.tensor.shape[0]:
                index_tensor = self.tensor[self.index]
                self.index += 1
                return index_tensor
            else:
                raise StopIteration

    def __add__(self, other: Union['Tensor', int ,float]) -> 'Tensor':
        if isinstance(other,Tensor):
            return Tensor.from_c_tensor(lib.tensor_add(self.c_tensor,other.c_tensor))
        else:
            return Tensor.from_c_tensor(lib.tensor_add_scalar(self.c_tensor, float(other)))

    def __sub__(self, other: Union['Tensor', int ,float]) -> 'Tensor':
        if isinstance(other,Tensor):
            return Tensor.from_c_tensor(lib.tensor_subtract(self.c_tensor,other.c_tensor))
        else:
            return Tensor.from_c_tensor(lib.tensor_subtract_scalar(self.c_tensor, float(other)))

    def __mul__(self, other: Union['Tensor', int ,float]) -> 'Tensor':
        if isinstance(other,Tensor):
            return Tensor.from_c_tensor(lib.tensor_multiply(self.c_tensor,other.c_tensor))
        else:
            return Tensor.from_c_tensor(lib.tensor_multiply_scalar(self.c_tensor, float(other)))

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return Tensor.from_c_tensor(lib.tensor_dot(self.c_tensor, other.c_tensor))

    def __setitem__(self, s: slice, o: Iterable[_T]) -> None:
        raise NotImplementedError("Tensor assignment not implemented")

    def insert(self, index: int, value: _T) -> None:
        raise NotImplementedError("Tensor insertion not implemented")

    def __delitem__(self, i: int) -> None:
        raise NotImplementedError("Tensor deletion not implemented")

    def __call__(self):
        return f"Tensor: ({self.flatten}, tensor dims: {self.dims}, shape: {self.shape})"

    def __str__(self):
        return self.__call__()

    def __repr__(self):
        return self.__call__()

    def __del__(self):
        if hasattr(self, 'c_tensor') and self.c_tensor and self.c_tensor != ffi.NULL:
            try:
                lib.tensor_free(self.c_tensor)
            except:
                pass  # Ignore errors during cleanup
            self.c_tensor = None
