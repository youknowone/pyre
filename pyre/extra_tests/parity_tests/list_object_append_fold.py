# CPython-suite gap: list tests do not trace object-strategy append transitions.
# parity-tests reason: this guards pyre's JIT list fold and GC write path.

# Object-strategy `list.append`, run hot enough to be traced.
#
# The append fold descends the real `w_list_append` body.  Its Object arm
# stores a GC ref, so unlike the unboxed Integer/Float arms it first runs the
# list write barrier through `prepare_list_ref_store`, which returns the value
# at its post-safepoint address.  Getting that wrong stores a stale pointer, so
# every case below reads back what it appended and folds it into a checksum.
#
# The fold only fires while the backing block has spare capacity; a full block
# side-exits to the resizing push.  Each loop therefore crosses several
# reallocation boundaries, and the interleaved `pop` / `insert` / slot
# assignment keep exercising the same block from the non-folded side.

N = 400


def check(got, want, label):
    assert got == want, "%s: %r != %r" % (label, got, want)


# ── grow from empty, read every element back ──
def append_str(n):
    xs = []
    i = 0
    while i < n:
        xs.append("e%d" % (i % 7))
        i = i + 1
    total = 0
    for s in xs:
        total = total + int(s[1:])
    return total


def append_tuple(n):
    xs = []
    i = 0
    while i < n:
        xs.append((i, i + 1))
        i = i + 1
    total = 0
    for a, b in xs:
        total = total + b - a
    return total


def append_none(n):
    xs = []
    i = 0
    while i < n:
        xs.append(None)
        i = i + 1
    return len(xs) + sum(1 for x in xs if x is None)


# ── a sliding window: the block never grows past the bound, so the fold's
# spare-capacity guard holds while `pop(0)` keeps rewriting the same slots ──
def append_pop_window(n):
    xs = []
    marker = "m"
    i = 0
    acc = 0
    while i < n:
        xs.append(marker)
        if len(xs) > 32:
            acc = acc + len(xs.pop(0))
        i = i + 1
    return acc * 1000 + len(xs)


# ── a heterogeneous list: the Object strategy is the only one that can hold
# it, and the appended element's type changes every iteration ──
def append_mixed(n):
    xs = []
    i = 0
    while i < n:
        r = i % 4
        if r == 0:
            xs.append(i)
        elif r == 1:
            xs.append(str(i))
        elif r == 2:
            xs.append(None)
        else:
            xs.append((i,))
        i = i + 1
    ints = 0
    strs = 0
    nones = 0
    tups = 0
    for x in xs:
        if x is None:
            nones = nones + 1
        elif isinstance(x, tuple):
            tups = tups + x[0] % 2
        elif isinstance(x, str):
            strs = strs + len(x)
        else:
            ints = ints + 1
    return ints * 1000000 + strs * 10000 + nones * 100 + tups


# ── an int list that de-specialises mid-loop: the first non-int append
# switches the storage to Object, so the same call site folds through two
# different arms ──
def append_despecialise(n):
    xs = []
    i = 0
    while i < n:
        if i == n // 2:
            xs.append("wall")
        else:
            xs.append(i)
        i = i + 1
    total = 0
    for x in xs:
        if isinstance(x, str):
            total = total + len(x) * 100000
        else:
            total = total + x
    return total


# ── a store into an already-appended slot, so the barrier runs on both the
# append and the assignment ──
def append_then_store(n):
    xs = []
    i = 0
    while i < n:
        xs.append("a")
        xs[len(xs) - 1] = "b%d" % (i % 5)
        i = i + 1
    total = 0
    for s in xs:
        total = total + int(s[1:])
    return total


# ── an insert in the middle keeps the block shifting under the fold ──
def append_and_insert(n):
    xs = []
    i = 0
    while i < n:
        xs.append("t")
        if i % 16 == 0:
            xs.insert(len(xs) // 2, "i")
        i = i + 1
    return len(xs) * 100 + sum(1 for s in xs if s == "i")


# ── two lists alternating at one call site: a receiver mix-up cross-checks ──
def append_two_receivers(n):
    a = []
    b = []
    i = 0
    while i < n:
        target = a if i % 2 == 0 else b
        target.append("x%d" % i)
        i = i + 1
    return len(a) * 1000 + len(b) + int(a[0][1:]) + int(b[0][1:])


def run():
    check(append_str(N), sum(i % 7 for i in range(N)), "append_str")
    check(append_tuple(N), N, "append_tuple")
    check(append_none(N), 2 * N, "append_none")
    check(append_pop_window(N), (N - 32) * 1000 + 32, "append_pop_window")
    check(append_mixed(N), expected_mixed(), "append_mixed")
    check(append_despecialise(N), sum(range(N)) - N // 2 + 400000, "append_despecialise")
    check(append_then_store(N), sum(i % 5 for i in range(N)), "append_then_store")
    check(append_and_insert(N), (N + 25) * 100 + 25, "append_and_insert")
    check(append_two_receivers(N), 200 * 1000 + 200 + 0 + 1, "append_two_receivers")


# `append_mixed`'s checksum is written out rather than recomputed so a wrong
# element cannot be cancelled by a matching wrong expectation.
def expected_mixed():
    ints = 0
    strs = 0
    nones = 0
    tups = 0
    for i in range(N):
        r = i % 4
        if r == 0:
            ints = ints + 1
        elif r == 1:
            strs = strs + len(str(i))
        elif r == 2:
            nones = nones + 1
        else:
            tups = tups + i % 2
    return ints * 1000000 + strs * 10000 + nones * 100 + tups


check(append_mixed(N), expected_mixed(), "append_mixed")
run()

# A second pass on already-warm code: every loop above has been traced by now,
# so this run executes the compiled form rather than building it.
run()

print("OK")
