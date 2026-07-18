"""Int dict strategy must not unwrap application-level int subclasses.

PyPy's ``IntDictStrategy.is_correct_type`` delegates to
``listobject.is_plain_int1``.  An int subclass therefore remains in object
storage, preserving both its identity and its per-instance state when dict
keys are reused by set operations.
"""


class HashCountingInt(int):
    def __init__(self, *args):
        self.hash_count = 0

    def __hash__(self):
        self.hash_count += 1
        return int.__hash__(self)


keys = [HashCountingInt(i) for i in range(10)]
d = dict.fromkeys(keys)

assert list(d) == keys
assert all(type(key) is HashCountingInt for key in d)
assert sum(key.hash_count for key in d) == 10

s = set(d)
assert sum(key.hash_count for key in d) == 10
s.difference(d)
assert sum(key.hash_count for key in d) == 10
s.symmetric_difference_update(d)
assert sum(key.hash_count for key in d) == 10

d2 = dict.fromkeys(set(d))
assert list(d2) == keys
assert sum(key.hash_count for key in d) == 10

print("OK")
