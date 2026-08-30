# pyre-check: max-pypy-ratio=5.2
# Ubuntu run 33279264115: 2.1-2.6x; the ceiling is twice the slowest,
# rounded up to one decimal place.
# A module-scope LOAD_NAME whose name misses the module dict resolves through
# the frame's builtin module.  The builtins cell folds under the module dict's
# version? (so a later global binding shadows the builtin) and the builtins
# dict's own version?.  The second loop proves that invalidation is seen.
# Output verified against CPython and PyPy.

N = 90000000
M = 400000
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
