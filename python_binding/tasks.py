from cffi import FFI
import os

ffi = FFI()

# Use relative path or environment variable for better portability
dll_path = os.environ.get('ML_LIB_PATH', "C:\\Users\\keyna\\source\\repos\\machine_learning_library\\x64\\Debug\\machine_learning_library.dll")
header_path = os.environ.get('ML_HEADER_PATH', "C:\\Users\\keyna\\source\\repos\\machine_learning_library\\include\\Cnet.h")

try:
    lib = ffi.dlopen(dll_path)
except OSError as e:
    print(f"Failed to load library: {e}")
    raise

try:
    with open(header_path, 'r') as f:
        header_content = f.read()
        # Remove any #include statements and preprocessor directives that CFFI can't handle
        lines = header_content.split('\n')
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith('#') and stripped:
                clean_lines.append(line)
        clean_header = '\n'.join(clean_lines)
        ffi.cdef(clean_header, override=True)
except FileNotFoundError as e:
    print(f"Header file not found: {e}")
    raise
except Exception as e:
    print(f"Error parsing header: {e}")
    raise

__all__ = ["ffi", "lib"]
print("c library works")