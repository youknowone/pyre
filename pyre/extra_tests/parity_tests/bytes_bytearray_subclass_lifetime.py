# CPython-suite gap: builtin-subclass tests omit finalizer lifetime for these types.
# parity-tests reason: this guards PyPy-style allocation and moving-GC ownership.

import gc


# bytesobject.py:descr_new and bytearrayobject.py:new_bytearray allocate strict
# subtypes through space.allocate_instance, so both participate in the user
# finalizer queue.
finalized = []


class FinalizingBytes(bytes):
    def __del__(self):
        finalized.append(("bytes", bytes(self)))


class FinalizingBytearray(bytearray):
    def __del__(self):
        finalized.append(("bytearray", bytes(self)))


bytes_obj = FinalizingBytes(b"abc")
bytearray_obj = FinalizingBytearray(b"xyz")
assert type(bytes(bytes_obj)) is bytes
assert type(bytearray(bytearray_obj)) is bytearray
del bytes_obj
del bytearray_obj
gc.collect()

assert sorted(finalized) == [("bytearray", b"xyz"), ("bytes", b"abc")]

print("OK")
