"""Python 3.14 memoryview.tobytes(order) conversion semantics."""

data = bytearray(range(6))
view = memoryview(data).cast("B", (2, 3))

assert view.tobytes() == b"\x00\x01\x02\x03\x04\x05"
assert view.tobytes(None) == b"\x00\x01\x02\x03\x04\x05"
assert view.tobytes("C") == b"\x00\x01\x02\x03\x04\x05"
assert view.tobytes(order="C") == b"\x00\x01\x02\x03\x04\x05"
assert view.tobytes("F") == b"\x00\x03\x01\x04\x02\x05"
assert view.tobytes(order="F") == b"\x00\x03\x01\x04\x02\x05"
assert view.tobytes("A") == b"\x00\x01\x02\x03\x04\x05"

wide = memoryview(bytearray(range(8))).cast("H", (2, 2))
assert wide.tobytes("C") == b"\x00\x01\x02\x03\x04\x05\x06\x07"
assert wide.tobytes("F") == b"\x00\x01\x04\x05\x02\x03\x06\x07"

strided = memoryview(data)[::2]
assert strided.tobytes("C") == b"\x00\x02\x04"
assert strided.tobytes("F") == b"\x00\x02\x04"
assert strided.tobytes("A") == b"\x00\x02\x04"


class Order(str):
    pass


assert view.tobytes(Order("F")) == b"\x00\x03\x01\x04\x02\x05"

for bad in ("", "c", "X"):
    try:
        view.tobytes(bad)
    except ValueError as error:
        assert str(error) == "order must be 'C', 'F' or 'A'"
    else:
        raise AssertionError(bad)

try:
    view.tobytes(1)
except TypeError as error:
    assert str(error) == "tobytes() argument 'order' must be str or None, not int"
else:
    raise AssertionError("integer order accepted")

try:
    view.tobytes("C", order="F")
except TypeError as error:
    assert str(error) == "tobytes() takes at most 1 argument (2 given)"
else:
    raise AssertionError("duplicate order accepted")

print("OK")
