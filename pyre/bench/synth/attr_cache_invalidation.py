# pyre-check: max-pypy-ratio=64
N = 100000


class C:
    def __init__(self, v):
        self.x = v


def read_all(objs, rounds):
    total = 0
    i = 0
    while i < rounds:
        for o in objs:
            total = total + o.x
        i = i + 1
    return total


def main():
    objs = [C(k) for k in range(4)]
    # Phase 1: monomorphic instance-attribute reads warm and hit the
    # LOAD_ATTR mapdict cache (same class, same map across all four objs).
    total = read_all(objs, N)
    # Phase 2: shadow `x` with a class-level data descriptor.  Assigning to
    # the class bumps its version_tag, which must invalidate the cached
    # instance-dict entry so the property getter wins over the instance dict.
    C.x = property(lambda self: 7)
    total = total + read_all(objs, N)
    print(total)


main()
