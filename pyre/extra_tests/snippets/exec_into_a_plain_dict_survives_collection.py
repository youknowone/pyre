# pyre-check: gate=1
"""A frame whose globals are not a module's `__dict__` still collects cleanly.

`exec(code, {...})` hands the frame a plain dict, which is an ordinary
collectable object rather than one of the module dicts the interpreter keeps
non-moving.  Everything the tracer then reads out of that frame and bakes as a
jitcode constant is therefore an address the collector is free to change, and a
constant pool that records an address once has no way to hear about the move.

The body below is the minimised pair that reaches it: a loop polymorphic enough
to compile, and a function object built inside the compiled frame.  Both halves
are load-bearing -- the loop over `seq` alone, or a module-level `def` passed in
place of the `lambda`, leaves the constant on the non-moving side.

The failure is not at any line here.  The program runs to completion and prints
both lines; the abort comes later, in the first collection that seeds its roots
from the pool, and it names an address whose object died two collections ago.

Which is also why a green run of this file is NOT on its own evidence that the
pool is rooted -- whether the collector reaches the stale slot before the
program ends depends on where the nursery happens to be.  The instrument that
answers directly is `PYRE_PROBE14=1`, which reports every pool slot the walk
forwards; on this body it reports one.
"""

SOURCE = '''
N = 50000


class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v


def call(fn):
    return fn()


def main():
    seq = "abcdefghij"
    lst = list(range(10))
    tup = tuple(range(10))
    acc = 0
    i = 0
    while i < N:
        a = Idx(i % 5)
        b = Idx(i % 5 + 3)
        acc = acc + len(seq[a:b]) + len(lst[a:b]) + len(tup[a:b])
        i = i + 1
    assert acc == 450000, acc
    assert call(lambda: 1) == 1


main()
'''

code = compile(SOURCE, "<foreign globals>", "exec")
exec(code, {"__name__": "__main__"})
print("OK")
