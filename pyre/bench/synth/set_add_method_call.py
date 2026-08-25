# pyre-check: max-pypy-ratio=30
# pyre-check: spec-folds=set_add_method
# The ceiling is twice the slowest of the three backends measured on one
# darwin host -- 12.0x dynasm, 12.3x cranelift, 14.2x wasm -- rounded up.
# The same binary at the same commit has read this fixture anywhere from 11.5x
# to 16.1x depending on what else the host was running, so the doubling is
# what absorbs the runner rather than slack for a regression.
#
# What it gates is therefore a gross regression, not this arm: doubling
# `hot_add` alone moves the sum by only ~28%, which no ceiling wide enough to
# survive that spread can catch. The pypy denominator here is ~0.14s, under
# three times `FLOOR_GATE_MIN_BASELINE_S`, thin enough that the ratio reads
# coarsely.
# `spec-folds` above is what gates the arm: an arm that stops firing reads
# exactly like one nobody wrote a leg for.
#
# The two `hot_add*` legs are deliberately the bulk of the time -- 52% of the
# fixture's dynasm execution (28% + 24%), with no other leg over 13%.
# `hot_add_user_hash` ran 300k Python-level `__hash__` calls at first and
# took 53% of the fixture, which would have made the ceiling a gate on
# user-dunder dispatch rather than on anything this file is about.
# `s.add(x)` and a set comprehension name one operation -- `pyopcode.py SET_ADD`
# is `space.call_method(w_set, 'add', w_value)` -- but they recorded two
# different residuals. The comprehension's SET_ADD lowers to the direct
# `set_add` store; the method call arrived as a generic `bh_call_fn`, which
# re-reads the bound method's function, rejects keywords and rebuilds the
# argument vector on every iteration. `try_walker_specialize_set_add_method`
# pins the callable to the `set.add` builtin and guards the receiver, then
# substitutes the direct store. `hot_add` is what that buys, and
# `hot_add_comprehension` is the spelling it is brought level with; a
# regression separates the two legs.
#
# The substitution is not a fold: the insert stays a MayForce residual because
# hashing the element can run a user `__hash__`. `hot_add_user_hash` is that
# case -- the element's `__hash__` and `__eq__` are Python, so the call really
# does force -- and `hot_add_raising_hash` walks the exception channel out of
# the substituted residual, which has to reach the `except` in the loop body
# exactly as the generic call did.
#
# The three remaining legs pin what the arm may and may not swallow, and are
# here because a wrong answer there is silently wrong code rather than a slow
# loop. `hot_add_subclass` MUST be substituted: the receiver guard is the
# builtin's own `require_set_receiver` predicate, an `ob_type` layout test that
# a `set` subclass shares, and an inherited `add` runs the identical body.
# `hot_add_override` MUST NOT be, and is the leg that says so -- its `add`
# stores a different value, so bypassing it changes `len`. `hot_add_unbound`
# calls the descriptor with an explicit receiver, which is one argument more
# than the arm accepts.


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
    while k < 10:
        total += hot_add(700000)
        total += hot_add_comprehension(700000)
        total += hot_add_growing(30000)
        total += hot_add_user_hash(5000)
        total += hot_add_raising_hash(3000)
        total += hot_add_subclass(30000)
        total += hot_add_override(30000)
        total += hot_add_unbound(30000)
        k += 1
    print(total)


main()
