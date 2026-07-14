from testutils import TestFailingBool, TestFailingIter, assert_raises

assert any([True])
assert not any([False])
assert not any([])
assert any([True, TestFailingBool()])


def any_stops_after_true():
    yield 1
    raise RuntimeError("unreachable")


assert any(any_stops_after_true())

assert_raises(RuntimeError, lambda: any(TestFailingIter()))
assert_raises(RuntimeError, lambda: any([TestFailingBool()]))
