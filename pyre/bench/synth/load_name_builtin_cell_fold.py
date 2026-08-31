# pyre-check: spec-folds=builtin_len
# A module-scope LOAD_NAME whose name misses the module dict resolves through
# the frame's builtin module.  The builtins cell folds under the module dict's
# version? (so a later global binding shadows the builtin) and the builtins
# dict's own version?.  The second loop proves that invalidation is seen: the
# census reads `builtin_len` consulted twice and fired once, the decline being
# the shadowed call.
#
# No pypy ratio gate: what this loop costs pypy varies by a multiple across the
# hosts while pyre holds steady, so no single ceiling covers the span and still
# derives a floor beneath it.  The census decides instead, and does not move
# with the host.  Output verified against CPython and PyPy.

N = 100000
M = 10000
s = "xx"

total = 0
for i in range(N):
    total = total + len(s)
print(total)

len = lambda x: 100
shadowed = 0
for i in range(M):
    shadowed = shadowed + len(s)
print(shadowed)
