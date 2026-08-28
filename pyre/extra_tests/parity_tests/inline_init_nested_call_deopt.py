# CPython-suite gap: no suite test deopts inside a helper called by a hot `__init__`.
# parity-tests reason: this guards the resume chain's SHAPE when the constructor
# continuation is not the innermost paused level.

"""A class call returns the instance when the deopt happens one frame deeper.

`inline_init_deopt_result.py` deopts inside `__init__` itself, so the level
that discards `__init__`'s result sits directly beneath the frame that
resumes.  Here `__init__` calls a helper and the guard fails inside THAT, so
the chain is `caller -> descr_call -> __init__ -> helper`.  The discarding
level is no longer adjacent to the resumed frame, which is the arrangement
that catches a continuation recorded at a fixed offset from the innermost
callee rather than at its own depth.

A chain that misplaces it answers `None` from the CALL, or delivers the
helper's return where the instance belongs, so the checks read the call's
result rather than only its side effects.
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


def classify(n):
    # The arm taken on the rare iterations is not the traced one, so the guard
    # that fails is inside this helper, one frame below `__init__`.
    if n < 0:
        return "neg", -n
    return "pos", n


class Boxed:
    def __init__(self, n):
        self.tag, self.magnitude = classify(n)
        self.n = n


total = 0
rare_seen = 0
for i in range(ROUNDS):
    v = -i if i % RARE == RARE - 1 else i
    b = Boxed(v)
    # The call's result, not a side effect: a misplaced continuation answers
    # `None` here, and one recorded a frame too low answers the helper's tuple.
    assert isinstance(b, Boxed), (i, b)
    assert b.n == v, (i, b.n, v)
    assert b.tag == ("neg" if v < 0 else "pos"), (i, b.tag, v)
    assert b.magnitude == abs(v), (i, b.magnitude, v)
    if v < 0:
        rare_seen += 1
    total += b.n

assert rare_seen == ROUNDS // RARE, (rare_seen, ROUNDS // RARE)
assert total == sum(-i if i % RARE == RARE - 1 else i for i in range(ROUNDS)), total


def maybe_bad(n):
    # Only ever reached after the loop is compiled, so the `TypeError` has to
    # come out of the resumed chain rather than out of tracing — and it is
    # raised by the level ABOVE the frame that returned the offending value.
    if n < 0:
        return n
    return None


class BadOnRarePath:
    def __init__(self, n):
        self.n = n
        return maybe_bad(n)


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
