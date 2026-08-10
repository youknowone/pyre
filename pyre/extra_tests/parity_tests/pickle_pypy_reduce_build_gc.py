# CPython-suite gap: CPython cannot emit PyPy's private packed-list pickle opcodes.
# parity-tests reason: this guards PyPy pickle compatibility and moving-GC roots.

"""PyPy `_pickle` parity for argument expansion and BUILD."""

import pickle
import sys


IS_CPYTHON = sys.implementation.name == "cpython"


if not IS_CPYTHON:
    # PyPy's homogeneous-list extension uses private GLOBAL sentinels followed
    # by a packed length-prefixed byte blob and REDUCE. CPython has no
    # `pypy._builtin` module.
    ascii_blob = b"\x03\x00\x00\x00\x01a\x02bc\x00"
    bytes_blob = b"\x03\x00\x00\x00\x01a\x02\xff\x00\x00"
    assert pickle.loads(
        b"\x80\x04cpypy._builtin\n_ascii_list_unpickle\nC"
        + bytes((len(ascii_blob),))
        + ascii_blob
        + b"\x85R."
    ) == ["a", "bc", ""]
    assert pickle.loads(
        b"\x80\x04cpypy._builtin\n_bytes_list_unpickle\nC"
        + bytes((len(bytes_blob),))
        + bytes_blob
        + b"\x85R."
    ) == [b"a", b"\xff\x00", b""]
    for name in (b"_ascii_list_unpickle", b"_bytes_list_unpickle"):
        try:
            pickle.loads(
                b"\x80\x04cpypy._builtin\n"
                + name
                + b"\nC\x04\x01\x00\x00\x00\x85R."
            )
        except ValueError as exc:
            assert "truncated" in str(exc)
        else:
            raise AssertionError("truncated packed list payload was accepted")

print("OK")
