# pyre-check: pypy-diverges: pypy3's EmptySetStrategy.remove answers without hashing, so set().remove([]) is KeyError and set().discard([]) is None there
# CPython-suite gap: `test_set` asserts TypeError for an unhashable `remove`
# and `discard` only on a populated set, so it never reaches the empty set.
#
# An empty set hashes the key before answering remove, discard and
# membership, the same as a populated one.
#
# parity-tests reason: the empty arm silently answered instead of raising.


def outcome(fn, arg):
    try:
        return repr(fn(arg))
    except Exception as e:
        return type(e).__name__


for s in (set(), {1}, {"a"}):
    for name in ("remove", "discard", "__contains__"):
        got = outcome(getattr(s, name), [])
        assert got == "TypeError", (len(s), name, got)

s = {1}
s.remove(1)
assert outcome(s.remove, []) == "TypeError"
assert outcome(s.discard, {}) == "TypeError"

# A set argument is retried as a frozenset and stays a miss.
assert outcome(set().remove, {1}) == "KeyError"
assert outcome(set().discard, {1}) == "None"

print("OK")
