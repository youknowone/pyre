# cpyext-fixture: cpyext_cffi_pypy
# cpyext-expect: cpyext-cffi-pypy-ok

# PyPy generated-CFFI extension loading.

import sys
import cpyext_cffi_pypy as module

assert type(module.ffi).__module__ == '_cffi_backend'
assert type(module.ffi).__name__ == 'FFI'
assert type(module.lib).__module__ == '_cffi_backend'
assert type(module.lib).__name__ == 'Lib'
assert sys.modules['cpyext_cffi_pypy.lib'] is module.lib
assert module.__file__.endswith('.so')
print('cpyext-cffi-pypy-ok')
