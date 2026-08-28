# CPython-suite gap: the suite does not run a hot user-defined __next__ whose
# FOR_ITER sits inside a try range in the same frame.
# parity-tests reason: FOR_ITER's materialized catch routes a non-StopIteration
# exception to the Python handler covering its PC, so the handler edge, the
# multi-clause match order and the finally path must all survive that rewiring.

advances = []


class Raising:
    def __init__(self, limit, raise_at, exc):
        self.index = 0
        self.limit = limit
        self.raise_at = raise_at
        self.exc = exc

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        self.index += 1
        advances.append(self.index)
        if self.index == self.raise_at:
            raise self.exc("boom")
        return self.index


# The loop is inside the try, so the raise takes FOR_ITER's handler edge
# rather than leaving the frame.
def consume_caught(limit, raise_at, exc):
    total = 0
    caught = None
    try:
        for value in Raising(limit, raise_at, exc):
            total += value
    except ValueError as e:
        caught = ("ValueError", str(e))
    except TypeError as e:
        caught = ("TypeError", str(e))
    return total, caught


# Hot with no exception at all: the try range must not disturb exhaustion.
for _ in range(12):
    total, caught = consume_caught(2000, 0, ValueError)
    assert total == 2000 * 2001 // 2, total
    assert caught is None, caught

# The second except clause must win when the first does not match, which
# exercises the handler's own match chain downstream of FOR_ITER's edge.
advances.clear()
total, caught = consume_caught(2000, 1500, TypeError)
assert caught == ("TypeError", "boom"), caught
assert total == 1499 * 1500 // 2, total
assert len(advances) == 1500, len(advances)

advances.clear()
total, caught = consume_caught(2000, 1500, ValueError)
assert caught == ("ValueError", "boom"), caught
assert len(advances) == 1500, len(advances)


# A finally between the loop and the handler must still run exactly once.
def consume_finally(limit, raise_at):
    marks = []
    try:
        for _ in Raising(limit, raise_at, ValueError):
            pass
    except ValueError:
        marks.append("caught")
    finally:
        marks.append("finally")
    return marks


for _ in range(12):
    assert consume_finally(1200, 0) == ["finally"]

assert consume_finally(1200, 900) == ["caught", "finally"]


# An unhandled kind inside the try must leave the frame, not be swallowed by
# the handler edge.
def consume_escapes(limit, raise_at):
    try:
        for _ in Raising(limit, raise_at, KeyError):
            pass
    except ValueError:
        return "wrong"
    return "no-raise"


for _ in range(12):
    assert consume_escapes(1200, 0) == "no-raise"

escaped = 0
try:
    consume_escapes(1200, 700)
except KeyError:
    escaped = 1

assert escaped == 1

print("OK")
