import struct


uninitialized = struct.Struct.__new__(struct.Struct)
try:
    uninitialized.__sizeof__()
except RuntimeError as exc:
    assert str(exc) == "Struct object is not initialized"
else:
    raise AssertionError("half-initialized Struct.__sizeof__ did not fail")


expected = {
    "": 88,
    "i": 120,
    "10i": 120,
    "2s": 120,
    "0i": 88,
    "  i  h": 152,
    "@i": 120,
    "100x": 88,
    "2p": 120,
}
for format_string, size in expected.items():
    assert struct.Struct(format_string).__sizeof__() == size


class WithSlot(struct.Struct):
    __slots__ = ("extra",)


assert WithSlot("").__sizeof__() == 96
