# `min()`/`max()` with `key=`: the callback re-enters the interpreter, so the
# running best item and best key must survive a collection that happens inside
# it. Both live in shadow-stack slots and are re-read from those slots at every
# use, which is only correct if the surrounding bookkeeping still observes the
# order `min_max` specifies -- one key computed and compared at a time, strict
# comparison so equal keys keep the first-seen extremum. These cases pin that
# observable behaviour: the tie rule for both directions, the exact call order
# and count of the callback, a key= that allocates every iteration, error
# propagation out of the compare and out of the callback itself, and the
# `default=` path that returns before any of the rooting starts. Output
# verified against CPython/PyPy.
N = 20000


def tie_keeps_first(n):
    # A later equal key must never displace the incumbent, in either direction.
    pairs = [(i, i & 3) for i in range(n)]
    lo = min(pairs, key=lambda p: p[1])
    hi = max(pairs, key=lambda p: p[1])
    return lo, hi


def call_order(n):
    # Exactly one call per item, in iteration order: the key of an item is
    # computed and compared immediately rather than all keys up front.
    seen = []

    def spy(i):
        seen.append(i)
        return i & 7

    lo = min(range(n), key=spy)
    return lo, len(seen), seen[0], seen[-1]


def allocating_key(n):
    # Each callback allocates a fresh string, so the best key is a young object
    # that must stay reachable across the following callbacks.
    items = ("abcdefgh" + str(i) for i in range(n))
    shortest = min(items, key=lambda s: len(s))
    items = ("abcdefgh" + str(i) for i in range(n))
    longest = max(items, key=lambda s: len(s))
    return shortest, longest


def nested_key(n):
    # The callback itself calls min(), so a second rooting bracket is open
    # while the outer one is live.
    rows = [[(i + j) & 15 for j in range(4)] for i in range(n)]
    return min(rows, key=lambda r: min(r)), max(rows, key=lambda r: min(r))


def errors(n):
    out = []
    try:
        min([1, "a"])
    except TypeError:
        out.append("compare TypeError")
    try:
        min([])
    except ValueError:
        out.append("empty ValueError")

    def boom(x):
        if x == n - 1:
            raise KeyError(x)
        return x

    try:
        min(range(n), key=boom)
    except KeyError as e:
        out.append("key KeyError %d" % e.args[0])
    return out


def defaults():
    return min([], default="d"), max([], default="d"), min([5], default="d")


def no_key(n):
    return min(range(n)), max(range(n)), min(3, 1, 2), max(3, 1, 2), min([7])


print(tie_keeps_first(N))
print(call_order(N))
print(allocating_key(N))
print(nested_key(N // 8))
print(errors(N))
print(defaults())
print(no_key(N))
