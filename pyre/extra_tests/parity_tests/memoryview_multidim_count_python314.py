"""Python 3.14 keeps multidimensional memoryview count unsupported."""


mv = memoryview(b"1234").cast("B", (2, 2))
try:
    mv.count(50)
except NotImplementedError as exc:
    assert str(exc) == "multi-dimensional sub-views are not implemented"
else:
    raise AssertionError("multidimensional memoryview.count() must fail")
print("OK")
