# pyre-check: spec-folds=builtin_len
# pyre-check: skip-cpython
# This fixture states no pypy ceiling, and the absence is the finding rather
# than an allowance.  The loop body folds to integer arithmetic on both
# interpreters, so the ratio compares two JITs' code for a counting loop; this
# interpreter's own time on it is steady from one runner to the next, and
# pypy's is not.  Across the fleet the baseline moves far enough that the
# readings span the whole factor `perf_gate_floor` divides a ceiling by, so
# every ceiling wide enough for the slowest runner derives a floor above what
# the fastest reads, and every ceiling under that floor fails the slowest.
# There is no value that holds both bounds, and the spread is the denominator's.
# What the fixture asserts is the fold census above and its jit-stats.
# A module-scope LOAD_NAME whose name misses the module dict resolves through
# the frame's builtin module.  The builtins cell folds under the module dict's
# version? (so a later global binding shadows the builtin) and the builtins
# dict's own version?.  The second loop proves that invalidation is seen: the
# census reads `builtin_len` consulted twice and fired once, the decline being
# the shadowed call.

N = 225000000
M = 4000000
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
