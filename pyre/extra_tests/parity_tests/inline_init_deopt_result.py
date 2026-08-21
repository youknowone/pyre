# CPython-suite gap: no suite test deopts inside a constructor hot enough to inline.
# parity-tests reason: this guards what a class call returns after a deopt in `__init__`.

"""A class call returns the instance, even when `__init__` deopts.

An inlined `__init__` is a resume level of its own, so a guard inside it leaves
the blackhole running `__init__` forward to its `return`.  What that return
carries is `None`; what the CALL has to produce is the instance.  The level
that reconciles the two is `descr_call`'s tail, and if it is missing or wrong
the call quietly answers `None` instead of raising anything — so the checks
below read the result rather than just its side effects.

`__init__` returning a non-`None` value is a `TypeError`, and that check lives
in the same place, so a body that only returns non-`None` on a rare (deopting)
path has to raise it there too.
"""

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=1,function_threshold=1")

ROUNDS = 4000
# Rare enough that the loop is long compiled before the second arm is first
# taken, so that arm is reached through a guard failure rather than by tracing.
RARE = 997


class Boxed:
    def __init__(self, n):
        # The arm taken on the rare iterations is not the traced one.
        if n < 0:
            self.tag = "neg"
        else:
            self.tag = "pos"
        self.n = n


total = 0
rare_seen = 0
for i in range(ROUNDS):
    v = -i if i % RARE == RARE - 1 else i
    b = Boxed(v)
    # The call's result, not a side effect: a lost discard answers `None` here.
    assert isinstance(b, Boxed), (i, b)
    assert b.n == v, (i, b.n, v)
    assert b.tag == ("neg" if v < 0 else "pos"), (i, b.tag, v)
    if v < 0:
        rare_seen += 1
    total += b.n

assert rare_seen == ROUNDS // RARE, (rare_seen, ROUNDS // RARE)
assert total == sum(-i if i % RARE == RARE - 1 else i for i in range(ROUNDS)), total


class BadOnRarePath:
    def __init__(self, n):
        self.n = n
        if n < 0:
            # Only ever reached after the loop is compiled, so the `TypeError`
            # has to come out of the resumed chain rather than out of tracing.
            return n
        return None


raised = 0
built = 0
for i in range(ROUNDS):
    v = -i if i % RARE == RARE - 1 else i
    try:
        obj = BadOnRarePath(v)
    except TypeError:
        raised += 1
    else:
        assert isinstance(obj, BadOnRarePath), (i, obj)
        assert obj.n == v, (i, obj.n, v)
        built += 1

# `n == 0` is not negative, so iteration 0 builds; every rare iteration after
# it raises.
assert raised == ROUNDS // RARE, (raised, ROUNDS // RARE)
assert built == ROUNDS - raised, (built, ROUNDS, raised)

print("OK")
