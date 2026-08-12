# CPython-suite gap: traceback tests walk a chain as soon as it is caught and
# never retain many chains across forced collections.
# parity-tests reason: a JIT-emitted traceback node is nursery-resident where
# the host constructor's is not, so only pyre can read one back after a move.

import gc

# A JIT-emitted traceback node is nursery-resident, so a minor collection
# copies it and rewrites the slots that name it. Walking a chain as soon as it
# is caught never observes that, because no collection intervenes; retaining
# the chains and churning the nursery first is what reads a node back through
# an address the collector has since moved.
#
# The chain shape is fixed (module -> outer -> middle -> inner), so a node read
# through a stale address shows up as a wrong `co_name`, a wrong line, or a
# short chain rather than as a crash.

KEPT = 600
CHURN = 12
EXPECTED_NAMES = ["<module>", "outer", "middle", "inner"]


def inner(i):
    raise KeyError(i)


def middle(i):
    inner(i)


def outer(i):
    middle(i)


def churn():
    total = 0
    for _ in range(CHURN):
        total += len([[] for _ in range(8)])
    return total


held = []
for i in range(KEPT):
    try:
        outer(i)
    except KeyError as e:
        held.append((i, e, e.__traceback__))
    churn()

gc.collect()

for i, exc, tb in held:
    names, linenos = [], []
    node = tb
    while node is not None:
        names.append(node.tb_frame.f_code.co_name)
        linenos.append(node.tb_lineno)
        node = node.tb_next
    assert names == EXPECTED_NAMES, (i, names)
    assert all(0 < lineno < 1000 for lineno in linenos), (i, linenos)
    assert exc.args == (i,), (i, exc.args)

# The innermost node must still name the raising line, not the helper's `def`.
first_linenos = []
node = held[0][2]
while node is not None:
    first_linenos.append(node.tb_lineno)
    node = node.tb_next
assert len(set(first_linenos)) == len(first_linenos), first_linenos

print(f"kept={len(held)} depth={len(EXPECTED_NAMES)}")
print("OK")
