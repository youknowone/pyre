# CPython-suite gap: test_ctypes does not exercise pyre's callback libffi CIF.
# parity-tests reason: callback argument/result ABI and nested function pointers.

from ctypes import CFUNCTYPE, Structure, c_double, c_int, c_longlong, c_void_p, cast


class Pair(Structure):
    _fields_ = [("x", c_int), ("y", c_double)]


# An aggregate argument must retain its field layout.  Describing these bytes
# as a pointer or an all-byte struct changes Win64/SysV register classification.
PAIR_CALLBACK = CFUNCTYPE(c_double, Pair)
pair_callback = PAIR_CALLBACK(lambda pair: pair.x + pair.y)
assert pair_callback(Pair(7, 0.5)) == 7.5

# The callback result slot has the declared scalar width, not pointer width.
I64_CALLBACK = CFUNCTYPE(c_longlong, c_longlong)
i64_callback = I64_CALLBACK(lambda value: value + 0x100000000)
assert i64_callback(9) == 0x100000009

# Function-pointer parameters arrive as callable `_CFuncPtr` instances whose
# `_ptr` and CData buffer both contain the foreign address.
INNER = CFUNCTYPE(c_int, c_int)
APPLY = CFUNCTYPE(c_int, INNER, c_int)
inner = INNER(lambda value: value + 1)
apply = APPLY(lambda function, value: function(value) * 2)
assert apply(inner, 20) == 42

assert cast(pair_callback, c_void_p).value
print("OK")
