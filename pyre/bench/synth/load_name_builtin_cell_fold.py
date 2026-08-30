# pyre-check: max-pypy-ratio=2.8
# Run 33300212586, dynasm and cranelift over all three hosts: 0.5-2.7x.  That
# span is 5.4x against a floor divisor of 6, so the ceiling and the floor it
# derives leave 11% of room between them; 2.8 splits it, sitting 1.04x above
# the slowest reading with a 0.47x floor 1.07x below the fastest.  What moves
# is pypy's own baseline -- 0.11s on ubuntu against 0.21s on macos while pyre
# holds 0.18-0.29s -- and windows already declines it as under
# FLOOR_GATE_MIN_BASELINE_S.  Sizing the fixture so that baseline clears the
# minimum on every host is what would give this gate margin to spare.
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
