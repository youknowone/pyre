# pyre-check: selfcheck
# pyre-check: selfcheck-loops=1
# Self-checking guard that a foreign frame's `f_lasti` keeps its residual.
#
# The fold's whole licence is that the walk KNOWS the coordinate of the frame
# being read: the portal frame's own pc, or an inlined callee's.  A frame the
# walk does not own carries a coordinate only its own writer knows -- a
# suspended generator's, a traceback node's -- so the read has to stay the
# residual getter that goes to the heap.  This fixture is the negative half of
# the fold's identity gate: nothing here may be answered from the walk's pc.
#
# The suspended generator is the sharper of the two, because its answer is
# STABLE across the loop the way a folded constant would be, and wrong by a
# fold that keyed on the reading frame instead of the read one.  The traceback
# node has its own oracle beside it: `tb_lasti` is resolved by the recorder at
# raise time, so `tb_frame.f_lasti == tb_lasti` for a frame the exception has
# already left.
import sys

N = 20000

FIRST = sys._getframe().f_lineno


def gen():
    i = 0
    while True:
        yield i                                              # +6
        i += 1


def raises(i):
    raise ValueError(i)


def main():
    g = gen()
    next(g)
    gframe = g.gi_frame
    suspended = set()
    unwound = set()
    total = 0
    for i in range(N):
        suspended.add((gframe.f_code, gframe.f_lasti))
        try:
            raises(i)
        except ValueError as e:
            tb = e.__traceback__.tb_next
            unwound.add((tb.tb_frame.f_lasti, tb.tb_lasti))
        total += i

    if len(suspended) != 1:
        print(f"FAIL suspended generator f_lasti diverged: {sorted(v for _, v in suspended)}")
        return 1
    code, lasti = next(iter(suspended))
    if lasti < 0 or lasti % 2 != 0:
        print(f"FAIL suspended generator f_lasti not an even byte offset: {lasti}")
        return 1
    row = list(code.co_positions())[lasti // 2][0]
    if row is None or row - FIRST != 6:
        print(f"FAIL suspended generator f_lasti={lasti} names line {row} not +6")
        return 1
    if len(unwound) != 1:
        print(f"FAIL unwound frame f_lasti diverged: {sorted(unwound)}")
        return 1
    frame_lasti, tb_lasti = next(iter(unwound))
    if frame_lasti != tb_lasti:
        print(f"FAIL unwound frame f_lasti={frame_lasti} != tb_lasti={tb_lasti}")
        return 1
    if total != sum(range(N)):
        print(f"FAIL dropped iteration: total={total}")
        return 1
    print("PASS foreign frame f_lasti stays residual")
    return 0


sys.exit(main())
