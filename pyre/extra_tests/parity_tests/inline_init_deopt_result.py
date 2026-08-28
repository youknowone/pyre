# CPython-suite gap: no suite test deopts inside a constructor hot enough to inline.
# parity-tests reason: this guards what a class call returns after a deopt in `__init__`.

"""A rare, resumed non-None `__init__` result still raises `TypeError`.

The ordinary inlined-constructor result path is owned by
``bench/synth/type_call_inline_init_branch_deopt.py``.  This residual case pins
the other arm of ``typeobject.py descr_call``: the non-None result check must
still run when the result only appears after a guard failure.
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
