from testutils import TestFailingBool, TestFailingIter, assert_raises

assert all([True])
assert not all([False])
assert all([])
assert not all([False, TestFailingBool()])


def all_stops_after_false():
    yield 0
    raise RuntimeError("unreachable")


assert not all(all_stops_after_false())

assert_raises(RuntimeError, all, TestFailingIter())
assert_raises(RuntimeError, all, [TestFailingBool()])
