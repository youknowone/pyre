# pyre-check: skip-backends=cranelift
# cranelift runs the core build, which has no `pyre-module` and so no `math`.
# The loop over `cdf(invcdf(p))` reads each lambda's closure cell, a heap
# short box whose receiver is another short box.  The loop's own close adds
# that box to the LABEL through `ExtendedShortPreambleBuilder`; a bridge back
# onto the loop must carry it too, or its JUMP is one arg short of the LABEL.
from statistics import NormalDist


def make_pair():
    nd = NormalDist(0.0, 1.0)
    cdf = lambda t: nd.cdf(t)
    invcdf = lambda t: nd.inv_cdf(t)
    return cdf, invcdf


def check(first, second, places):
    if first == second:
        return 0
    if round(abs(first - second), places) == 0:
        return 0
    return 1


def run(cdf, invcdf):
    parr = [i / 1000 + 5 / 10000 for i in range(1000)]
    bad = 0
    for _ in range(2):
        for p in parr:
            bad += check(cdf(invcdf(p)), p, 11)
    return bad


cdf, invcdf = make_pair()
print("bad =", run(cdf, invcdf))
