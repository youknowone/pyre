assert list(reversed(range(5))) == [4, 3, 2, 1, 0]

l = [5, 4, 3, 2, 1]
assert list(reversed(l)) == [1, 2, 3, 4, 5]

# `reversed` is a lazy iterator (`W_ReversedIterator`): construction performs
# no item access, and it walks the sequence backward by index.  Sequences
# without their own `__reversed__` (tuple, str) yield this generic type.
assert type(reversed((1, 2, 3))).__name__ == "reversed"
assert list(reversed((1, 2, 3))) == [3, 2, 1]
assert list(reversed("abc")) == ["c", "b", "a"]

# __length_hint__ tracks the elements not yet produced.
it = reversed([10, 20, 30])
assert it.__length_hint__() == 3
next(it)
assert it.__length_hint__() == 2

# __reduce__ exposes (reversed, (sequence,), index) so a partially consumed
# iterator resumes at the right position; __setstate__ restores the cursor.
it = reversed([1, 2, 3])
recon, args, state = it.__reduce__()
assert args == ([1, 2, 3],) and state == 2, (args, state)
next(it)
assert it.__reduce__()[2] == 1
# Reconstruct + setstate (what copy / pickle perform).
it2 = recon(*args)
it2.__setstate__(1)
assert list(it2) == [2, 1]

# Laziness: constructing a reversed iterator must not call __getitem__.
class Seq:
    def __init__(self):
        self.gets = []

    def __len__(self):
        return 3

    def __getitem__(self, i):
        self.gets.append(i)
        return i * 10


s = Seq()
r = reversed(s)
assert s.gets == []
assert list(r) == [20, 10, 0]
assert s.gets == [2, 1, 0]
