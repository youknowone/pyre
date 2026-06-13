import io

# io.UnsupportedOperation is a real (OSError, ValueError) exception class
# with a settable __module__ (io.py:78 assigns it), usable in raise/except.
assert isinstance(io.UnsupportedOperation, type), io.UnsupportedOperation
assert issubclass(io.UnsupportedOperation, OSError)
assert issubclass(io.UnsupportedOperation, ValueError)
assert io.UnsupportedOperation.__module__ == "io", io.UnsupportedOperation.__module__
try:
    raise io.UnsupportedOperation("nope")
except ValueError as e:
    assert isinstance(e, io.UnsupportedOperation)
    assert isinstance(e, OSError)

# io.BlockingIOError resolves to the builtin BlockingIOError.
assert io.BlockingIOError is BlockingIOError

# io.BytesIO is a working in-memory binary stream.
b = io.BytesIO()
b.write(b"hello"); b.write(b" world")
assert b.getvalue() == b"hello world", b.getvalue()
assert b.tell() == 11
b.seek(0)
assert b.read(5) == b"hello"
assert b.readline() == b" world"
b2 = io.BytesIO(b"abc")
assert b2.read() == b"abc"
assert b2.read() == b""
b3 = io.BytesIO()
b3.write(b"line1\nline2\n")
b3.seek(0)
assert list(b3) == [b"line1\n", b"line2\n"]

# Negative-bound slice assignment (STORE_SLICE) — used by pickle's
# _Unpickler.load_tuple3 (`self.stack[-3:] = [...]`).
lst = [1, 2, 3, 4, 5]
lst[-3:] = [(lst[-3], lst[-2], lst[-1])]
assert lst == [1, 2, (3, 4, 5)], lst
lst2 = [1, 2, 3, 4, 5]
lst2[-3:-1] = [9]
assert lst2 == [1, 2, 9, 5], lst2

# import pickle (pure-Python module) now succeeds: the _io exception
# fixes let `import io` import, and the chain through `import re` works.
import pickle
assert pickle is not None

print("pickle_io_prereq OK")
