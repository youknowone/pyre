# pyre-check: max-pypy-ratio=170
# pypy's execution-only time here is a few hundredths of a second, so most runs
# clamp the baseline to EXEC_TIME_FLOOR_S and decline the gate outright; the
# ceiling is only ever applied on the runs that land just above the clamp,
# where the denominator carries its own magnitude as error.  Over the 38 CI
# jobs of 2026-09-03 those runs read 23.9x-117.4x on unchanged code.  Giving
# pypy enough work to measure is not open here: pyre runs about 70x slower, so
# a baseline over FLOOR_GATE_MIN_BASELINE_S would put this fixture past 18s.
# 170 clears the widest reading by 45%.  It is an envelope on a noisy
# denominator, not a claim about how fast this shape is.
N = 200000

METH = '\udc81'   # lone surrogate naming a method on the class
PROP = '\udc82'   # lone surrogate naming a property (data descriptor)
A1 = '\udc83'            # lone surrogate naming a per-instance attribute
A2 = '\udc84\udc85'      # multi-surrogate per-instance attribute name
A3 = 'ascii_attr'        # plain name stored alongside the surrogate nodes


def _meth(self):
    return 1


class P:
    pass


setattr(P, METH, _meth)
setattr(P, PROP, property(lambda self: 2))


def main():
    p = P()
    acc = 0
    i = 0
    # Surrogate-named attribute access through the full descriptor protocol
    # in a JIT-compiled hot loop: a non-data descriptor (function bound
    # through the type MRO) and a data descriptor (property __get__).
    while i < N:
        acc = acc + getattr(p, METH)()
        acc = acc + getattr(p, PROP)
        i = i + 1

    # Post-loop tail running in the already-compiled `main` frame:
    # per-instance surrogate-named attributes are stored as mapdict nodes
    # (keyed by their full WTF-8 name), interleaved with a plain-named one,
    # then read back, summed through __dict__, deleted and re-added.
    setattr(p, A1, 5)
    setattr(p, A2, 7)
    setattr(p, A3, 11)
    acc = acc + getattr(p, A1) + getattr(p, A2) + getattr(p, A3)
    acc = acc + sum(p.__dict__.values())
    acc = acc + (1 if A1 in p.__dict__ else 0)
    acc = acc + (1 if A2 in p.__dict__ else 0)
    delattr(p, A1)
    acc = acc + (0 if hasattr(p, A1) else 1)
    setattr(p, A1, 13)
    acc = acc + getattr(p, A1)
    print(acc)


main()
