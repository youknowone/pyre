# pyre-check: max-pypy-ratio=2.8
# pyre-check: skip-cpython
# Sized so pypy's own execution is a measurement on every host rather than a
# clock tick.  At the previous 90000000/400000 windows measured pypy at 0.059s
# of execution against a FLOOR_GATE_MIN_BASELINE_S of 0.156s there, so that
# host's floor gate declined the baseline outright, and pyre's own startup was
# a third of what remained to subtract on the other two.  Ten times the trip
# counts puts pypy at 0.59s on windows (3.8x that minimum), 0.97s on ubuntu and
# 1.94s on macos, and leaves the startup subtraction at 8-16% of a reading.
# cpython needs minutes at this size, which SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S
# already dropped on two of the three hosts.
#
# The size does not narrow the ratio itself: measured at one, two and four
# times the trip counts, pypy and pyre both scale linearly and the ratio held
# 1.60, 1.67, 1.66.  Run 33300212586 read dynasm and cranelift at 0.5-2.7x
# across the three hosts -- a 5.4x span against a floor divisor of six, so the
# ceiling and the floor it derives leave 11% of room between them, and 2.8
# splits it: 1.04x above the slowest reading, its 0.47x floor 1.07x below the
# fastest.  What moves is pypy, at 0.097s of execution on ubuntu against 0.194s
# on macos while pyre holds 0.11-0.19s.  A span that wide is the case
# PERF_GATE_FLOOR_DIVISOR names as one to investigate rather than absorb;
# carrying a fold census here would retire the ratio gate the way the fixtures
# around it did.
# A module-scope LOAD_NAME whose name misses the module dict resolves through
# the frame's builtin module.  The builtins cell folds under the module dict's
# version? (so a later global binding shadows the builtin) and the builtins
# dict's own version?.  The second loop proves that invalidation is seen.
# Output verified against CPython and PyPy.

N = 900000000
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
