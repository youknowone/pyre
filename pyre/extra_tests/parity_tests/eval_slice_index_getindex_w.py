# CPython-suite gap: slice-bound conversion is tested only through integer
# literals; int subclasses, bigint clamps, and the rewritten TypeError are not.
# parity-tests reason: `_eval_slice_index` must use `space.getindex_w` for
# exact int, bigint overflow clamp, int subclass, and missing `__index__`.

"""BINARY_SLICE bound conversion through space.getindex_w.

`sliceobject.py` `_eval_slice_index` calls `space.getindex_w`. Exact ints,
int subclasses, and values that overflow a machine word must slice the same
way; a bound with no `__index__` raises the rewritten TypeError.
"""


class MyInt(int):
    pass


def expect_typeerror(fn):
    try:
        fn()
    except TypeError as exc:
        assert str(exc) == (
            "slice indices must be integers or None or have an __index__ method"
        )
        return
    raise AssertionError("expected TypeError")


xs = [0, 1, 2, 3, 4]
assert xs[1:4] == [1, 2, 3]
huge = 10**100
assert xs[huge:] == []
assert xs[-huge:] == [0, 1, 2, 3, 4]
assert xs[MyInt(1) : MyInt(4)] == [1, 2, 3]
expect_typeerror(lambda: xs[1.5:2])
expect_typeerror(lambda: xs[object():2])

print("OK")
