# pyre-check: selfcheck
# A module-level `for` executed by `exec(src, globals, mapping)` whose locals
# are a non-dict mapping.  `journal_walker_namespace_write` records nothing
# for that namespace: `__setitem__` cannot be rolled back.  The assignment
# after the loop variable is therefore a body effect when the walk later
# aborts.  The call count must stay equal to the iterations.
import collections.abc

N = 2000


class NS(collections.abc.MutableMapping):
    def __init__(self):
        self.data = {}
        self.sets = 0

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.sets += 1
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


SRC = """
seen = []
n = 0
for i in range(N):
    n = n + 1
    seen.append(i)
"""

ns = NS()
exec(SRC, {"__builtins__": __builtins__, "range": range, "N": N}, ns)
# `seen`, `n` before the loop, then `i` and `n` once per iteration.
expected = 2 + 2 * N
if ns.sets != expected or ns["n"] != N or len(ns["seen"]) != N:
    raise SystemExit(f"sets={ns.sets} n={ns['n']} seen={len(ns['seen'])} expected_sets={expected}")
print(ns.sets)
print("PASS")
