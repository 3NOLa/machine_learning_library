from cffi import FFI

ffi = FFI()

lib = ffi.dlopen("C:\\Users\\keyna\\source\\repos\\machine_learning_library\\x64\\Debug\\machine_learning_library.dll")

with open("C:\\Users\\keyna\\source\\repos\\machine_learning_library\\include\\Cnet.h") as f:
    ffi.cdef(f.read(), override=True)

__all__ = ["ffi", "lib"]

l = lib.Sigmoid_function(10)
print(l)