from cffi import FFI
from pathlib import Path
import os

ffi = FFI()

directory_path = Path(__file__).parent.parent.parent
# write here the directory for the dll and header of the c library
dll_path = directory_path / 'x64' / 'Debug' / 'machine_learning_library.dll'
header_path = directory_path / 'include' / 'Cnet.h'

try:
    lib = ffi.dlopen(str(dll_path))
except OSError as e:
    print(f"Failed to load library: {e}")
    raise

try:
    with open(str(header_path), 'r') as f:
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