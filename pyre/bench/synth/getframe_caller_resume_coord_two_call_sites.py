# A callee reads its CALLER's resume coordinate -- `f_lineno` and `f_lasti` --
# from two DIFFERENT call sites in the same hot loop.
#
# `f_lineno` and `f_lasti` both resolve off the frame's `last_instr`, which
# compiled code does not store per opcode, so the value only reaches the frame
# if the force publishes it.  A single call site makes that unobservable: the
# caller's coordinate is then constant by construction and a frozen read is
# indistinguishable from a live one.  Calling from two lines makes the correct
# answer ALTERNATE, so a frame left holding one leg's coordinate collapses the
# two rows into one.
#
# The staleness this pins runs in the other direction too: a publish that is
# withdrawn too early -- put back at the residual's return instead of at walk
# end -- leaves the caller answering for the coordinate it held BEFORE the
# call, and the shape set gains an extra row rather than losing one.  Surveying
# every iteration into a set is what catches both, because the pre-compile
# iterations are correct and a miss appears as a changed row count.
#
# Measured, by putting that defect back in: dropping the `flushed` test from
# `LiveLastInstrGuard::drop`, so the guard always restores at the residual's
# return, prints
#     ([(0, 3), (0, 8), (1, 3), (1, 6)], [0, 0, 1, 1], 3)
# against this fixture's
#     ([(0, 8), (1, 6)], [0, 1], 2)
# -- the pre-call coordinate appears alongside the call-site one on BOTH legs,
# and the offsets' discrimination count rises with it.  Nothing else in
# bench/synth reads `f_lasti` at all, and only the traceback-frame fixture
# reads an `f_lineno`.
#
# `f_lasti` is a bytecode offset, so its absolute value is a property of the
# compiler and cannot be compared against the pypy oracle.  Only its
# DISCRIMINATION is portable: the two call sites must report two distinct
# offsets, one per leg.  `f_lineno` is a source coordinate and is compared
# directly, relative to `co_firstlineno` so edits above these functions do not
# move the expected values.
#
# Neither read goes through `.f_locals`: that getset forces the frame on its
# own, and a forcing read would mask exactly the staleness under test.
import sys

N = 30000


def inner():
    f = sys._getframe(1)
    return (f.f_lineno - f.f_code.co_firstlineno, f.f_lasti)


def outer(n):
    lines = set()
    offsets = set()
    i = 0
    while i < n:
        if i & 1:
            line, off = inner()
        else:
            line, off = inner()
        lines.add((i & 1, line))
        offsets.add((i & 1, off))
        i = i + 1
    legs = sorted(leg for leg, _ in offsets)
    distinct = len({off for _, off in offsets})
    return (sorted(lines), legs, distinct)


print(outer(N))
