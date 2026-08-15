# CPython-suite gap: test_ctypes does not exercise pyre's callback libffi CIF.
# parity-tests reason: callback argument/result ABI and nested function pointers.

from ctypes import (
    CFUNCTYPE,
    Structure,
    c_byte,
    c_double,
    c_int,
    c_longlong,
    c_short,
    c_ubyte,
    c_void_p,
    cast,
)


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

# A result narrower than the return word owns the whole word: its unwritten
# bytes are whatever the closure left there unless the slot is cleared first,
# and a negative value must still read back negative.
BYTE_CALLBACK = CFUNCTYPE(c_byte, c_byte)
byte_callback = BYTE_CALLBACK(lambda value: value)
assert byte_callback(-1) == -1
assert byte_callback(127) == 127

UBYTE_CALLBACK = CFUNCTYPE(c_ubyte, c_ubyte)
ubyte_callback = UBYTE_CALLBACK(lambda value: value)
assert ubyte_callback(255) == 255

SHORT_CALLBACK = CFUNCTYPE(c_short, c_short)
short_callback = SHORT_CALLBACK(lambda value: value)
assert short_callback(-2) == -2
assert short_callback(0x7FFF) == 0x7FFF

INT_CALLBACK = CFUNCTYPE(c_int, c_int)
int_callback = INT_CALLBACK(lambda value: value)
assert int_callback(-3) == -3
assert int_callback(0x7FFFFFFF) == 0x7FFFFFFF

# Function-pointer parameters arrive as callable `_CFuncPtr` instances holding
# the foreign address, not as the integer a scalar decoder would produce.  A
# failed assertion inside a callback has nowhere to propagate, so it is
# reported and the call returns zero — which the outer comparison then catches.
INNER = CFUNCTYPE(c_int, c_int)
APPLY = CFUNCTYPE(c_int, INNER, c_int)
inner = INNER(lambda value: value + 1)
inner_address = cast(inner, c_void_p).value


def apply_body(function, value):
    assert cast(function, c_void_p).value == inner_address
    return function(value) * 2


apply = APPLY(apply_body)
assert apply(inner, 20) == 42

assert cast(pair_callback, c_void_p).value
print("OK")
