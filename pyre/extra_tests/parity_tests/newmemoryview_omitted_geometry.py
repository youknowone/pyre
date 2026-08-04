"""`__pypy__.newmemoryview` distinguishes an omitted geometry from `None`.

`interp_buffer.py:47 if w_strides` / `:56 if w_shape` are reference tests over
the host-level default, so an explicitly passed `None` is a supplied argument
that reaches `space.listview` and fails there.
"""

try:
    import __pypy__
except ImportError:  # the CPython reference runner has no __pypy__
    print("OK")
    raise SystemExit

source = memoryview(bytearray(b"abcdefgh"))

derived = __pypy__.newmemoryview(source, 1, "B")
assert derived.shape == (8,)
assert derived.strides == (1,)

supplied = __pypy__.newmemoryview(source, 2, "H", (2, 2), (4, 2))
assert supplied.shape == (2, 2)
assert supplied.strides == (4, 2)

for keyword in ("shape", "strides"):
    try:
        __pypy__.newmemoryview(source, 1, "B", **{keyword: None})
    except TypeError as error:
        assert str(error) == "'NoneType' object is not iterable", error
    else:
        raise AssertionError(f"{keyword}=None must not derive the geometry")

try:
    __pypy__.newmemoryview(source, 1, "B", None, None)
except TypeError as error:
    assert str(error) == "'NoneType' object is not iterable", error
else:
    raise AssertionError("positional None must not derive the geometry")

print("OK")
