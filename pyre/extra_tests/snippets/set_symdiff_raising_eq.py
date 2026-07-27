# symmetric_difference / ^ with a user __eq__/__hash__ that raises mid-probe.
# Exercises the JIT-residualised per-key contains/insert boundary: the parked
# exception must surface identically whether the merge loop runs interpreted or
# jitted.  Run under both the interpreter and the JIT (the harness warms the
# loop past the trace threshold).


class Boom:
    """Hashes into one bucket so every pair collides and __eq__ is forced."""

    def __init__(self, key, explode=False):
        self.key = key
        self.explode = explode

    def __hash__(self):
        return 7  # single bucket -> __eq__ runs on every probe

    def __eq__(self, other):
        if self.explode or getattr(other, "explode", False):
            raise ValueError("boom in __eq__")
        return self.key == other.key


def run_symdiff(a, b):
    return a ^ b


# --- baseline: no raise, plain-int fast path stays bit-exact ---------------
for _ in range(1000):
    assert {1, 2, 3} ^ {2, 3, 4} == {1, 4}
    assert {1, 2, 3}.symmetric_difference({2, 3, 4}) == {1, 4}
    assert set() ^ {1} == {1}
    assert {1} ^ set() == {1}


# --- object elements, no raise: hash collision forces __eq__ but succeeds ---
for _ in range(1000):
    left = {Boom(1), Boom(2), Boom(3)}
    right = {Boom(3), Boom(4)}
    result = run_symdiff(left, right)
    keys = sorted(e.key for e in result)
    assert keys == [1, 2, 4], keys


# --- raising __eq__ mid-probe: the ValueError must propagate identically ----
for _ in range(1000):
    left = {Boom(1), Boom(2)}
    right = {Boom(1, explode=True)}
    raised = False
    try:
        run_symdiff(left, right)
    except ValueError as exc:
        raised = True
        assert str(exc) == "boom in __eq__"
    assert raised, "symmetric_difference swallowed a raising __eq__"


# --- raising __eq__ on the second pass (walk this side) ---------------------
for _ in range(1000):
    left = {Boom(9, explode=True)}
    right = {Boom(5)}
    raised = False
    try:
        run_symdiff(left, right)
    except ValueError as exc:
        raised = True
        assert str(exc) == "boom in __eq__"
    assert raised, "symmetric_difference swallowed a raising __eq__ on the second pass"


print("set_symdiff_raising_eq OK")
