# CPython-suite gap: dict tests do not trace Unicode-strategy changes at one site.
# parity-tests reason: this guards pyre's JIT strategy fold and side exit.

# `d[key]` on a str-keyed dict, run hot enough to be traced.
#
# The subscript fold pins the dict to the Unicode strategy and the key to the
# canonical `str` class, then records `rstr.ll_strhash` as an elidable call and
# the probe itself as the `rordereddict dict.lookup` oopspec.  The optimizer is
# allowed to drop a repeat of the same `(dict, key)` probe, so the entry index a
# compiled loop uses can come from an EARLIER iteration.  Every case below is
# built so a wrongly-retained index shows up as a wrong value rather than as a
# missing crash.

N = 400


def check(got, want, label):
    assert got == want, "%s: %r != %r" % (label, got, want)


# ── the plain shape: one loop-invariant dict, one loop-invariant key ──
def repeat_same_key(n):
    d = {"tag": 3, "other": 5}
    total = 0
    i = 0
    while i < n:
        total = total + d["tag"]
        i = i + 1
    return total


# ── keys that are EQUAL but not the same object.  The identity-keyed
# `dict.get` fold could never reach this; the hash+equality probe must ──
def fresh_key_each_iteration(n):
    d = {"tag": 7}
    total = 0
    i = 0
    while i < n:
        k = ("tagX")[:-1]
        total = total + d[k]
        i = i + 1
    return total


# ── the value under one key is rewritten every iteration.  A value-only
# overwrite keeps the entry index, so the fold may keep it — but the VALUE read
# must stay live, never folded to the recorded one ──
def value_overwritten(n):
    d = {"acc": 0}
    i = 0
    while i < n:
        d["acc"] = d["acc"] + i
        i = i + 1
    return d["acc"]


# ── the KEY SET changes under the loop.  Each insert can move the entry a
# retained index points at, so a stale index would read the wrong value ──
def key_set_grows(n):
    d = {"probe": 1}
    total = 0
    i = 0
    while i < n:
        d["k%d" % i] = i
        total = total + d["probe"]
        i = i + 1
    return total


# ── a delete compacts the table, renumbering every entry after the hole ──
def key_set_shrinks(n):
    d = {}
    i = 0
    while i < 64:
        d["k%d" % i] = i
        i = i + 1
    d["probe"] = 99
    total = 0
    i = 0
    while i < n:
        victim = "k%d" % (i % 64)
        if victim in d:
            del d[victim]
        total = total + d["probe"]
        i = i + 1
    return total


# ── two dicts alternating at one call site: a receiver mix-up cross-checks ──
def two_receivers(n):
    a = {"tag": 11}
    b = {"tag": 22}
    total = 0
    i = 0
    while i < n:
        total = total + (a if i % 2 == 0 else b)["tag"]
        i = i + 1
    return total


# ── the dict leaves the Unicode strategy mid-loop.  Storing a non-str key
# promotes it to the Object strategy, so the strategy guard must side-exit and
# the same call site must keep answering correctly afterwards ──
def strategy_switches(n):
    d = {"tag": 4}
    total = 0
    i = 0
    while i < n:
        if i == n // 2:
            d[17] = 100
        total = total + d["tag"]
        i = i + 1
    return total + d[17]


# ── a miss must raise KeyError, not answer a neighbouring entry ──
def miss_raises(n):
    d = {"present": 1}
    hits = 0
    misses = 0
    i = 0
    while i < n:
        try:
            hits = hits + d["present" if i % 3 else "absent"]
        except KeyError:
            misses = misses + 1
        i = i + 1
    return hits * 1000 + misses


# ── a dict SUBCLASS defines `__missing__`, so the fold must decline for it and
# the generic path must reach the override ──
class WithMissing(dict):
    def __missing__(self, key):
        return len(key)


def subclass_missing(n):
    d = WithMissing()
    d["here"] = 2
    total = 0
    i = 0
    while i < n:
        total = total + d["here"] + d["absent"]
        i = i + 1
    return total


# ── a str SUBCLASS key may override `__hash__`/`__eq__`, so it must not take
# the exact-str probe ──
class Shouty(str):
    def __hash__(self):
        return hash(str(self).lower())

    def __eq__(self, other):
        return str(self).lower() == str(other).lower()


def subclass_key(n):
    d = {"tag": 6}
    plain = 0
    shouty = 0
    i = 0
    while i < n:
        plain = plain + d["tag"]
        try:
            shouty = shouty + d[Shouty("TAG")]
        except KeyError:
            shouty = shouty - 1
        i = i + 1
    return plain * 100000 + shouty


def run():
    check(repeat_same_key(N), 3 * N, "repeat_same_key")
    check(fresh_key_each_iteration(N), 7 * N, "fresh_key_each_iteration")
    check(value_overwritten(N), sum(range(N)), "value_overwritten")
    check(key_set_grows(N), N, "key_set_grows")
    check(key_set_shrinks(N), 99 * N, "key_set_shrinks")
    check(two_receivers(N), (11 + 22) * (N // 2), "two_receivers")
    check(strategy_switches(N), 4 * N + 100, "strategy_switches")
    check(miss_raises(N), expected_miss(), "miss_raises")
    check(subclass_missing(N), (2 + 6) * N, "subclass_missing")
    check(subclass_key(N), expected_subclass_key(), "subclass_key")


# `miss_raises` / `subclass_key` checksums are written out from the same
# per-iteration rule rather than reusing the function, so a wrong answer cannot
# be cancelled by a matching wrong expectation.
def expected_miss():
    hits = 0
    misses = 0
    for i in range(N):
        if i % 3:
            hits = hits + 1
        else:
            misses = misses + 1
    return hits * 1000 + misses


def expected_subclass_key():
    # `Shouty("TAG")` hashes as "tag" and compares equal to it, so every
    # iteration finds the entry through the subclass's own protocol.
    return (6 * N) * 100000 + 6 * N


run()

# A second pass on already-warm code: every loop above has been traced by now,
# so this run executes the compiled form rather than building it.
run()

print("OK")
