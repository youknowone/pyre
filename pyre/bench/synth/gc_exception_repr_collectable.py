# pyre-check: no-cpython
import gc


empty = ValueError()
single = ValueError("x")
multiple = ValueError("x", 1)

empty_direct = BaseException.__repr__(empty)
empty_ordinary = repr(empty)
single_direct = BaseException.__repr__(single)
single_ordinary = repr(single)
multiple_direct = BaseException.__repr__(multiple)
multiple_ordinary = repr(multiple)

assert empty_direct == empty_ordinary == "ValueError()"
assert single_direct == single_ordinary == "ValueError('x')"
assert multiple_direct == multiple_ordinary == "ValueError('x', 1)"

results = (
    empty_direct,
    empty_ordinary,
    single_direct,
    single_ordinary,
    multiple_direct,
    multiple_ordinary,
)
for result in results:
    assert any(obj is result for obj in gc.get_objects())

print("exception repr results are collectable")
