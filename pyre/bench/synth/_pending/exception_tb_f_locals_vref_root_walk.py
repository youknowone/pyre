# Reading `f_locals` off the OUTERMOST traceback node — the catching frame —
# leaves a JIT virtual ref in that frame's `locals_cells_stack_w`. A minor
# collection triggered by a nursery allocation from compiled code then walks the
# slot as a GC root. A vref's leading word is the `JIT_VIRTUAL_REF_VTABLE` magic
# rather than a PyObject `ob_type`, so a root walker that hands the slot to the
# raw exception walk dereferences the magic as a type pointer and takes a
# SIGSEGV. Reading the innermost node instead is clean.
#
# The collection has to land while the vref is on the stack, which is why the
# loop is long: on cranelift the crash is deterministic from roughly the 6000th
# iteration (2000 and 4000 stay clean). dynasm and the plain interpreter never
# reach that GC point on this shape, so this costs all three backends a run to
# gate one of them.
#
# NOT YET A SUITE FIXTURE — run it as `<pyre> thisfile.py 15000 head`.
#
# Run with NO arguments it aborts on cranelift for an unrelated, still-open
# reason: `GC BUG ... site=minor_varsize_item_target`. That is not this file's
# subject and it is not fixed by the walker guard — it reproduces identically
# with the guard reverted. The trigger is allocation layout, not the values:
# this same file with `15000 head` on the command line is clean, and taking the
# identical numbers from the `else` defaults aborts. Wrapping the defaults to
# defeat constant folding (`int("15000")`, `len(sys.argv)` arithmetic) does not
# help, which is what rules the folding explanation out.
#
# It can move up to `pyre/bench/synth/` once that abort is fixed.
#
# Expected output: 1 [(('drive', ('e', 'k', 'seen')), 'mid')]

import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
WHICH = sys.argv[2] if len(sys.argv) > 2 else "head"


def mid(i):
    raise ValueError("boom")


def locs(tb):
    out = []
    idx = 0
    while tb is not None:
        f = tb.tb_frame
        want = (
            WHICH == "all"
            or (WHICH == "head" and idx == 0)
            or (WHICH == "tail" and tb.tb_next is None)
        )
        if want:
            out.append((f.f_code.co_name, tuple(sorted(f.f_locals))))
        else:
            out.append(f.f_code.co_name)
        tb = tb.tb_next
        idx += 1
    return tuple(out)


def drive():
    seen = set()
    k = 0
    while k < N:
        try:
            mid(k)
        except ValueError as e:
            seen.add(locs(e.__traceback__))
            e.__traceback__ = None
        k += 1
    return sorted(seen)


r = drive()
print(len(r), r)
