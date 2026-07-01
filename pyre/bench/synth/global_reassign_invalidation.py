N = 200000

G = 100


def read_global():
    return G


# Module-level range-driven warmup loop.  Calling `read_global` from here
# invokes it as a residual call from the loop trace, so it compiles at
# *function entry* with its LOAD_GLOBAL const-folding the module dict's
# quasi-immutable cell for `G` to 100.
s = 0
for i in range(N):
    s = s + read_global()

# Reassign the module global.  The write bumps the module dict version,
# which must invalidate the compiled function-entry trace's baked cell
# value via the registered version watcher; otherwise the compiled
# `read_global` keeps returning the stale 100.
G = 200

# The first post-reassign call goes through the compiled function-entry
# trace path and must observe the fresh value, not the baked 100.
first = read_global()

# A second warmup keeps the value read fresh throughout, not only on the
# first deopt.
t = 0
for i in range(1000):
    t = t + read_global()

print(first, t)
