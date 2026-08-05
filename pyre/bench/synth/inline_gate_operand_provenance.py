# pyre-check: max-pypy-ratio=164
# pyre-check: min-pypy-ratio=10
# The FOR_ITER inline gate admits a callee whose only unproven residual is a
# `BINARY_OP` it expects the walker to specialize away.  That expectation rests
# on `args_all_exact_*`, which describes the callee's INCOMING ARGUMENTS — so it
# only holds for a binop whose operands are those arguments.
#
# The first three drivers put a numeric subclass with a side-effecting dunder
# where the argument check says nothing: a module global, a local copied from
# one, an attribute of a global.  The specialization declines on the subclass,
# so the residual survives an admission that promised it would not, and the
# inline sub-walk's deopt resumes at the caller's CALL boundary and re-executes
# the whole callee.
#
# `len(LOG)` counts dunder entries, so a doubled replay shows up there even when
# the arithmetic result survives it.  It does NOT move today: with the operand
# proof disabled these bodies are admitted, yet the effect still lands once,
# because the shapes that realize the replay are caught downstream.  So this
# pins behaviour and documents the shapes rather than reproducing a live bug —
# the last driver is the load-bearing one, holding the exemption open for the
# case it is actually meant to cover.
N = 50000

LOG = []


class Sneaky(int):
    def __sub__(self, other):
        LOG.append(1)
        return int.__sub__(self, other)

    def __add__(self, other):
        LOG.append(1)
        return int.__add__(self, other)

    def __mul__(self, other):
        LOG.append(1)
        return int.__mul__(self, other)


G = Sneaky(5)


# The operand is a module global: `LoadGlobal` is a residual the body scan
# treats as a replay-safe READ, so it passes the scan while carrying a value
# nobody vouched for.
def global_operand_body(x):
    return G - x


def global_operand(n):
    s = 0
    for i in range(n):
        s += global_operand_body(i & 7)
    return s


# Reached through a local rather than straight off the global load, so the
# provenance has to survive a STORE_FAST / LOAD_FAST round trip rather than
# being visible in the one instruction.
def via_local_body(x):
    g = G
    return g + x


def via_local(n):
    s = 0
    for i in range(n):
        s += via_local_body(i & 7)
    return s


# The subclass is an attribute of a global container, so the operand arrives
# through an attribute load instead of a bare global load.
class Holder:
    pass


H = Holder()
H.value = Sneaky(3)


def attr_operand_body(x):
    return H.value * x


def attr_operand(n):
    s = 0
    for i in range(n):
        s += attr_operand_body(i & 7)
    return s


# Control: the same shape with both operands genuinely coming from the
# arguments.  This one the gate SHOULD keep admitting, so it pins that the
# provenance proof did not simply turn the whole exemption off.
def args_only_body(a, b):
    return a - b


def args_only(n):
    s = 0
    for i in range(n):
        s += args_only_body(i & 7, 1)
    return s


print(global_operand(N))
print(via_local(N))
print(attr_operand(N))
print(args_only(N))
print(len(LOG))
