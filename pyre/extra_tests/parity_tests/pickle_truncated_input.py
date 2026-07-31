"""Only a missing dispatch opcode is EOFError.

CPython's ``Modules/_pickle.c:6978-6981`` translates the short read performed
by the outer opcode loop to EOFError.  Once an opcode has been read, a short
payload remains UnpicklingError through ``bad_readline``.
"""

import io
import pickle


for data, expected in (
    (b"", EOFError),
    (b"N", EOFError),
    (b"J", pickle.UnpicklingError),
    (b"\x80", pickle.UnpicklingError),
):
    try:
        pickle.Unpickler(io.BytesIO(data)).load()
    except expected:
        pass
    else:
        raise AssertionError((data, expected))

print("OK")
