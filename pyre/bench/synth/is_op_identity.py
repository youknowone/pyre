# `IS_OP` reaches `ptr_eq` / `ptr_ne` instead of the generic COMPARE_OP
# residual.
#
# No `max-pypy-ratio`: this fixture exists to pin the fold's ANSWERS, and most
# of its time is the `semantics` half, which deliberately runs the declined
# residual path.  `goto_if_not_same_box` carries the fold's perf gate.
#
# The codewriter routes `is` / `is_not` through the same `compare_fn` residual
# as the six ordinary comparisons (tags 8 and 9), so before the walker fold
# every identity test in a loop cost a `CALL_MAY_FORCE` plus its
# `GUARD_NOT_FORCED`.  `space.is_w` (baseobjspace.py:833) is pointer identity
# for every class that does not override `is_w`, so those tests lower to a bare
# `ptr_eq` / `ptr_ne` — and a self-compare (`a is a`) answers at `is_w`'s
# opening `ptr::eq` whatever the class, so it folds to a constant with no op
# and no guard at all (FASTPATHS_SAME_BOXES, pyjitpl.py:326-336).
#
# `hot` holds only the shapes the fold accepts; each never-taken arm adds a
# huge sentinel, so a wrong predicate balloons the checksum.  `semantics` runs
# the seven classes that DO override `is_w` with a value comparison — int,
# float, complex, tuple, bytes, str, frozenset — where the fold must decline
# and leave the residual in place.  A `GuardClass` pins only the layout, and an
# `int` subclass instance shares `INT_TYPE` with a plain `int`, so folding
# those to `ptr_eq` would answer `False` where the interpreter answers `True`.
N = 200000


class Plain:
    pass


def hot():
    a = Plain()
    b = Plain()
    acc = 0
    i = 0
    while i < N:
        if a is a:  # same box: constant True
            acc += 1
        if a is not a:  # same box: constant False
            acc += 1000000
        if a is b:  # distinct instances: ptr_eq False
            acc += 1000000
        if a is not b:
            acc += 2
        if a is None:  # NoneType keeps pointer identity
            acc += 1000000
        if a is not None:
            acc += 4
        if b is Plain:  # instance vs its type object
            acc += 1000000
        i += 1
    return acc


def semantics(n):
    # Every operand is built from `n` so the compiler cannot fold the pair to
    # one constant: these must be two runtime objects of equal value.
    out = []
    out.append((n * 0 + 1000) is (n * 0 + 1000))  # int: equal by value
    out.append((n * 0.0 + 1.5) is (n * 0.0 + 1.5))  # float: equal by bits
    out.append((n * 0.0 + 0.0) is -(n * 0.0 + 0.0))  # float: 0.0 is not -0.0
    out.append(complex(n * 0, n * 0) is complex(n * 0, n * 0))  # complex
    out.append(tuple(range(n * 0)) is tuple(range(n * 0)))  # empty tuple
    out.append(tuple(range(n * 0 + 2)) is tuple(range(n * 0 + 2)))  # non-empty
    out.append(bytes(n * 0) is bytes(n * 0))  # empty bytes
    out.append(("a" * (n * 0 + 1)) is ("a" * (n * 0 + 1)))  # 1-char str
    out.append(("ab" * (n * 0 + 1)) is ("ab" * (n * 0 + 1)))  # 2-char str
    out.append(frozenset(range(n * 0)) is frozenset(range(n * 0)))  # empty
    out.append(True is (n < 1))  # bool singletons
    return out


def main():
    print(hot())
    # The value-comparing classes answer `is` differently under CPython and
    # PyPy (`1000 is 1000` is False there, True here), so the answers
    # themselves are not printable in a fixture check.py oracles against both.
    # What IS engine-independent, and what a wrongly-widened fold breaks, is
    # that the compiled answer matches the interpreted one: `first` is the
    # very first, interpreted call and `again` comes after the loop is hot.
    first = semantics(0)
    for _ in range(2000):
        again = semantics(0)
    print(again == first, len(first))


main()
