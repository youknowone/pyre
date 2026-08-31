# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=root:leaf
# The multi-frame escape flush publishes frame 0's LOCALS; this guards the
# operand stack published with them.
#
# A redirected frame field read from an inlined callee forces the CALLER's
# virtualizable, so the walk aborts as an inline sub-walk and a multi-frame
# blackhole image resumes the two frames. `sys._getframe(1)` only names the
# frame: the force sits at the field, where `rvirtualizable.py
# hook_access_field` places it, and `f_lasti` reads `last_instr` -- one of the
# five `virtualizable_gen.rs` declares. The `f_code` the caller-identity check
# reads is not a lever; its gateway carries no marker.
#
# The walk keeps that virtualizable symbolic, so nothing it pushed or popped
# ever reached the live frame's slot array. When the abort lands on a walk
# that crossed the inner `FOR_ITER`'s exhaust, the outer `FOR_ITER` has
# already run and `GET_ITER` has built a FRESH inner iterator that exists only
# in the walk: resuming without publishing the stack leaves the PREVIOUS
# pass's exhausted iterator on TOS, the inner loop ends one iteration in, and
# every later `i` runs one pass ahead of the `j` beside it. The values the
# body reads stay intact, which is why this reads as a loop-control error and
# not a lost item.
#
# Measured before the stack was published: `k=1041 i=209 j=0`, where `k` says
# the body is at `i=208 j=1` -- the first abort lands at the default hotness
# threshold and every pass after it is shifted. `loops_compiled` is 0 there and
# 1 here; what reaches the JIT either way is `leaf`'s root trace, which is what
# this declares.
#
# The same read in a `while` loop (`getframe_while_escaping_read_frame_identity`)
# escapes just as often and cannot show this: with no iterator on the operand
# stack there is nothing for a stale image to hold.
import sys

_gf = sys._getframe

N = 5
PASSES = 6000
wrong = []


def leaf(x):
    fr = _gf(1)
    _ = fr.f_lasti
    name = fr.f_code.co_name
    if name != "main":
        wrong.append(name)
    return x + 1


def main():
    total = 0
    k = 0
    for i in range(PASSES):
        for j in range(N):
            total = leaf(total)
            if k != i * N + j:
                print(
                    f"FAIL dropped iteration: k={k} i={i} j={j} "
                    f"expected i={k // N} j={k % N}"
                )
                raise SystemExit(1)
            k += 1
    if total != PASSES * N:
        print(f"FAIL short run: total={total} expected {PASSES * N}")
        raise SystemExit(1)
    if wrong:
        print(f"FAIL caller identity: {wrong[:4]}")
        raise SystemExit(1)
    print(f"PASS total={total}")


main()
