# CPython-suite gap: list tests do not trace pop across strategy changes and guards.
# parity-tests reason: this guards pyre's eager JIT fold and rollback journal.

# Integer-strategy `list.pop()`, run hot enough to be traced.
#
# The pop fold descends the real `w_list_pop_end_inner` body, which is the
# `index == -1` half of `descr_pop` (`listobject.py`): a length read, an
# unboxed item read, a length store, and the wrap.  It fires only for the
# no-argument form on an Integer-strategy list; every other shape below has to
# decline cleanly and still answer correctly, which is what the checksums pin.
#
# The fold applies the pop eagerly while tracing and journals it, so a guard
# failure has to rewind both the length and the item.  The loops therefore
# alternate two receivers and re-read what they popped: a rewind that restores
# the length but not the value, or that restores them in the wrong order,
# corrupts a checksum here.

N = 400


def check(got, want, label):
    assert got == want, "%s: %r != %r" % (label, got, want)


# ── the folded shape: append then pop, hot, with a live result ──
def append_pop_live(n):
    xs = [0, 1, 2, 3, 4]
    total = 0
    i = 0
    while i < n:
        xs.append(i)
        total = total + xs.pop()
        i = i + 1
    return total * 100 + len(xs)


def expected_append_pop_live(n):
    return sum(range(n)) * 100 + 5


# ── the same shape with the result discarded, so the box is dead ──
def append_pop_dead(n):
    xs = [7, 8, 9]
    i = 0
    while i < n:
        xs.append(i)
        xs.pop()
        i = i + 1
    return len(xs) * 1000 + xs[0] * 100 + xs[2]


# ── two receivers alternating: a wrong rewind routes a value to the other list
def two_receivers(n):
    xs = [0]
    ys = [0]
    a = 0
    b = 0
    i = 0
    while i < n:
        xs.append(i)
        ys.append(-i)
        a = a + xs.pop()
        b = b + ys.pop()
        i = i + 1
    return a + b + len(xs) + len(ys)


# ── draining to empty, then one pop too many ──
def drain_then_overpop(n):
    xs = []
    i = 0
    while i < n:
        xs.append(i)
        i = i + 1
    total = 0
    while xs:
        total = total + xs.pop()
    try:
        xs.pop()
    except IndexError as e:
        return total * 10 + len(str(e).split()[0])
    return -1


# ── declines: the indexed form, and the non-Integer strategies ──
def pop_indexed(n):
    xs = list(range(n))
    total = 0
    i = 0
    while i < n // 2:
        total = total + xs.pop(0)
        i = i + 1
    return total * 100 + len(xs)


def pop_float(n):
    xs = [0.5]
    total = 0.0
    i = 0
    while i < n:
        xs.append(i + 0.5)
        total = total + xs.pop()
        i = i + 1
    return int(total * 2) * 10 + len(xs)


def pop_object(n):
    xs = ["seed"]
    total = 0
    i = 0
    while i < n:
        xs.append("e%d" % (i % 7))
        total = total + int(xs.pop()[1:])
        i = i + 1
    return total * 10 + len(xs)


# ── despecialisation mid-loop: an Integer list that becomes an Object one ──
def strategy_flip(n):
    xs = [0, 1, 2]
    total = 0
    i = 0
    while i < n:
        if i == n // 2:
            xs.append("x")
            total = total + len(xs.pop())
        else:
            xs.append(i)
            total = total + xs.pop()
        i = i + 1
    return total * 10 + len(xs)


def run():
    check(append_pop_live(N), expected_append_pop_live(N), "append_pop_live")
    check(append_pop_dead(N), 3 * 1000 + 7 * 100 + 9, "append_pop_dead")
    check(two_receivers(N), 2, "two_receivers")
    check(drain_then_overpop(N), sum(range(N)) * 10 + 3, "drain_then_overpop")
    check(pop_indexed(N), sum(range(N // 2)) * 100 + N - N // 2, "pop_indexed")
    check(pop_float(N), (sum(range(N)) * 2 + N) * 10 + 1, "pop_float")
    check(pop_object(N), sum(i % 7 for i in range(N)) * 10 + 1, "pop_object")
    check(strategy_flip(N), (sum(range(N)) - N // 2 + 1) * 10 + 3, "strategy_flip")


run()

# A second pass on already-warm code: every loop above has been traced by now,
# so this run executes the compiled form rather than building it.
run()

print("OK")
