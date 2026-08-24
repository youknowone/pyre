# pyre-check: gate=1
# `cast()` takes a pointer type, and a function-pointer type is one:
# `cffi/backend_ctypes.py CTypesGenericPtr._new_pointer_at` casts an address to
# the `CFUNCTYPE` it built for the signature.  The simple types it also takes
# are the ones `_ctypes.c cast_check_pointertype` names by code.
import ctypes

SIGNATURE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

buffer = ctypes.create_string_buffer(b"pyre")
address = ctypes.addressof(buffer)

# A function-pointer type.
function = ctypes.cast(address, SIGNATURE)
assert isinstance(function, SIGNATURE), type(function)
assert ctypes.cast(function, ctypes.c_void_p).value == address

# A pointer type built from a simple one, and the simple pointer-shaped types.
assert ctypes.cast(address, ctypes.POINTER(ctypes.c_int))
assert ctypes.cast(address, ctypes.c_void_p).value == address
assert ctypes.cast(address, ctypes.c_char_p).value == b"pyre"
assert ctypes.cast(address, ctypes.py_object)

# Not pointer-shaped, so refused.
for target in (ctypes.c_int, ctypes.c_double, int):
    try:
        ctypes.cast(address, target)
    except TypeError as error:
        assert "must be a pointer type" in str(error), error
    else:
        raise AssertionError("cast() accepted %r" % (target,))

print("done")
