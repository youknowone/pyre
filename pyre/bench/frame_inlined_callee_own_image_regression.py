# Self-checking regression guard for the image an INLINED CALLEE's own frame
# reports (registered via check.py run_selfcheck, NOT the synthetic suite).
#
# Sibling of `frame_lineno_mid_replay_regression`, which reads the coordinate
# through `sys._getframe(1)` from a callee whose caller is the portal — the
# frame the JIT treats as the virtualizable, and whose fields the walk keeps
# current.  The shape here is the other one: the frame handed out is the
# INLINED CALLEE's own, materialised through a `jit.virtual_ref` rather than
# being the virtualizable.  Such a frame is built with `last_instr = -1`
# (`pyre-jit-trace/src/helpers.rs`) and nothing updates it through the inlined
# body, and its locals region is not what the escape flush writes — that flush
# is keyed on the virtualizable.  So the image has to come from the force, and
# `offset2lineno(code, -1)` answers the `def` line when it does not.
#
# What routes these reads to a correct answer today is the abort:
# `sys._getframe` forces the virtualizable, `vable_after_residual_call` reads
# the cleared token as a callee escape, and the interpreter finishes the
# iteration.  That makes this guard load-bearing in an unusual way — it pins
# the ANSWER, not the mechanism, so a change that removes the abort in favour
# of a consumer-side force has to keep the answer.  Removing the two forces in
# `sys._getframe` and forcing from the `f_lineno` / `f_lasti` getsets instead
# left the whole 370-fixture corpus green while making `own_lineno` below
# report the `def` line, `own_locals` grow a second entry, and `own_combined`
# segfault.  None of the three had a fixture; that is why this one exists.
#
# Splitting each survey across several calls is what makes it a test, for the
# same reason the sibling guard does it: the loop compiles part-way through, so
# a set over the rounds holds the interpreted answer and the compiled one
# together, and a divergence appears as a SECOND element rather than a shifted
# single value.
#
# Offsets are relative to `co_firstlineno` so the expectations survive an edit
# above them.  It asserts rather than diffing against pypy3 because the shape
# it pins is a pyre-internal one; cpython and pypy3 both answer the same values
# and agree with `PYRE_NO_JIT=1`.
import sys

N = 4000
ROUNDS = 8


def own_lineno(x):                                   # +0
    f = sys._getframe(0)                             # +1
    return f.f_lineno - f.f_code.co_firstlineno      # +2


def own_locals(x):                                   # +0
    f = sys._getframe(0)                             # +1
    return tuple(sorted(f.f_locals))                 # +2


def own_combined(x):                                 # +0
    f = sys._getframe(0)                             # +1
    return (f.f_lineno - f.f_code.co_firstlineno, tuple(sorted(f.f_locals)), f.f_lasti >= 0)


def drive_lineno(n):
    seen = set()
    i = 0
    while i < n:
        seen.add(own_lineno(i))
        i += 1
    return seen


def drive_locals(n):
    seen = set()
    i = 0
    while i < n:
        seen.add(own_locals(i))
        i += 1
    return seen


def drive_combined(n):
    seen = set()
    i = 0
    while i < n:
        seen.add(own_combined(i))
        i += 1
    return seen


def main():
    each = N // ROUNDS
    failures = []

    def check(label, got, want):
        if got != want:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    seen = set()
    for _ in range(ROUNDS):
        seen |= drive_lineno(each)
    check("own_lineno", sorted(seen), [2])

    seen = set()
    for _ in range(ROUNDS):
        seen |= drive_locals(each)
    check("own_locals", sorted(seen), [("f", "x")])

    seen = set()
    for _ in range(ROUNDS):
        seen |= drive_combined(each)
    check("own_combined", sorted(seen), [(2, ("f", "x"), True)])

    if failures:
        for f in failures:
            print("FAIL", f)
        return 1
    print("PASS inlined-callee own frame image")
    return 0


sys.exit(main())
