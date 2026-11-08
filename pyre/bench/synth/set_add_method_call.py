# `s.add(x)` and a set comprehension name one operation -- `pyopcode.py SET_ADD`
# is `space.call_method(w_set, 'add', w_value)`. `hot_add` and
# `hot_add_comprehension` must agree on the result.
#
# `hot_add_user_hash` inserts keys whose `__hash__` and `__eq__` are Python.
# `hot_add_raising_hash` walks the exception channel out of the add and must
# reach the `except` in the loop body.
#
# `hot_add_subclass` inherits `add` on a `set` subclass. `hot_add_override`
# stores a different value, so bypassing it would change `len`.
# `hot_add_unbound` calls the descriptor with an explicit receiver.

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class Key:
    def __init__(self, v):
        self.v = v

    def __hash__(self):
        return self.v

    def __eq__(self, other):
        return isinstance(other, Key) and other.v == self.v


class BadHash:
    def __hash__(self):
        raise ValueError("unhashable on purpose")


class MySet(set):
    pass


class OverrideSet(set):
    def add(self, value):
        set.add(self, value & 15)


def hot_add(n):
    s = set()
    for i in range(n):
        s.add(i & 1023)
    return len(s)


def hot_add_comprehension(n):
    return len({i & 1023 for i in range(n)})


def hot_add_growing(n):
    # Distinct elements, so the backing storage rehashes underneath the
    # compiled loop instead of settling at a fixed occupancy.
    s = set()
    for i in range(n):
        s.add(i)
    return len(s)


def hot_add_user_hash(n):
    s = set()
    for i in range(n):
        s.add(Key(i & 255))
    return len(s)


def hot_add_raising_hash(n):
    s = set()
    bad = BadHash()
    caught = 0
    for i in range(n):
        s.add(i & 7)
        try:
            s.add(bad)
        except ValueError:
            caught += 1
    return len(s) * 100000 + caught


def hot_add_subclass(n):
    s = MySet()
    for i in range(n):
        s.add(i & 255)
    return len(s)


def hot_add_override(n):
    # 16 elements if the override ran, 256 if the arm bypassed it.
    s = OverrideSet()
    for i in range(n):
        s.add(i & 255)
    return len(s)


def hot_add_unbound(n):
    s = set()
    for i in range(n):
        set.add(s, i & 63)
    return len(s)


def main():
    total = 0
    k = 0
    # Every leg is independently hot in one pass; the fold census verifies the
    # method spelling without multiplying all eight workloads.
    while k < 1:
        total += hot_add(700000)
        total += hot_add_comprehension(700000)
        total += hot_add_growing(35000)
        total += hot_add_user_hash(7000)
        total += hot_add_raising_hash(3500)
        total += hot_add_subclass(35000)
        total += hot_add_override(35000)
        total += hot_add_unbound(35000)
        k += 1
    print(total)


main()
