import copy
import gc
import operator
import weakref


class Value:
    pass


a, b, c, d = [Value() for _ in range(4)]
original = weakref.WeakValueDictionary({a: b, c: d})
cloned = copy.copy(original)
del c, d
gc.collect()
gc.collect()
gc.collect()
assert len(original) == 1
assert len(cloned) == 1


left = []
left.append(left)
right = copy.deepcopy(left)
assert right is not left
assert right[0] is right
for comparison in (
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
):
    try:
        comparison(left, right)
    except RecursionError:
        pass
    else:
        raise AssertionError("recursive list comparison did not stop")


print("stdlib copy ok")

# Immutable tuples are returned directly, while deepcopy's memo bookkeeping
# for list/tuple members matches CPython 3.14.
immutable_tuple = ((1, 2), 3)
assert copy.deepcopy(immutable_tuple) is immutable_tuple

memo = {}
copy.deepcopy([1, 2, 3, 4], memo)
assert len(memo) == 2, memo

memo = {}
copy.deepcopy([(1, 2)], memo)
assert len(memo) == 2, memo
