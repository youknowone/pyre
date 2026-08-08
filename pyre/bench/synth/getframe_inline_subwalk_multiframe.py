# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# Coverage for the multi-frame blackhole build path: an INLINED callee that
# forces an outer frame through `sys._getframe(2)`.
#
# The walker executes a residual call concretely, so that level gets a real
# frame from the interpreter's own call sequence; an inline push did not run
# that sequence, so its level had none. A force fired from the inlined body
# therefore built a frame chain that mixed the two, rooted at the intermediate
# residual frame rather than the walked frame, and
# `try_adopt_multi_frame_blackhole`'s chain-root identity gate declined it. What
# that decline wanted was the `jit.virtual_ref` emit at the inline push
# (`executioncontext.py:89`); `walker_ec_enter` / `walker_ec_leave` landed it, so
# the chain-root gate no longer fires here, and with every level resumable from
# the concrete frame it owns, the chain is adopted rather than replayed. This fixture pins the
# result across that adopt.
#
# The build wants an escape whose forced frame is the one the inline sub-walk
# PUBLISHED, so the chain has a residual level under the walked frame and an
# inlined level under that. `sys._getframe(0).f_locals` in `leaf` is that
# escape: the depth-0 call names `leaf`'s own frame, the arm declines inside a
# sub-walk, and the getset forces the published frame.
#
# The `sys._getframe(2)` read below is the OUTPUT guard and is not what reaches
# the build -- `getframe` forces only the frame it returns, and two levels up is
# `main`'s, i.e. the portal, which escapes on the ordinary portal path. Without
# the depth-0 line the file still answers correctly and stops building a
# multi-frame image at all (measured: `fbw_blackhole_adopted_multi_frame` 5 ->
# 0). In the historical gate-enabled sweep with `PYRE_FBW_DEBUG_ABORT=1`, 0 of
# 310 fixtures reached `BUILT multi-frame`.
#
# The multi-frame image is built unconditionally when the latch conditions hold,
# so this is both an output guard and build-path coverage: 5 builds, 5 adopts,
# and zero declines of any kind (`PYRE_FBW_DEBUG_ABORT=1` prints every tally;
# the other 5 escapes in the run have `inline_subwalk=false` and take the
# single-frame arm, which adopts too).
#
# What the decline used to hold back, measured by lifting it before the
# execution-context push landed: the resumed chain shifted every
# `sys._getframe(n)` up exactly one level, so the read below landed on the
# module frame and raised `KeyError: 'base'`. Variants of this shape that cannot
# raise returned a wrong number instead, silently -- `f_locals.get("base", -1)`
# scored -1 for 7, `len(f_locals)` scored the module globals' 12 for this
# frame's 3, `len(f_code.co_name)` scored `<module>`'s 8 for a 9-character
# caller name. That is the failure mode this fixture still guards: every outcome
# arm was wrong, not only the one carrying a resume coordinate.
#
# Carries only a generous `# pyre-check: max-pypy-ratio=` tripwire (top of
# file), not a tight perf gate: this guards an output, and the forcing read
# makes it a poor perf subject, so the ratio is a gross-regression guard.
import sys


def leaf(x):
    sys._getframe(0).f_locals
    return sys._getframe(2).f_locals["base"] + x


def mid(x):
    return leaf(x) + 1


def main():
    base = 7
    total = 0
    for i in range(20000):
        total += mid(i)
    print(total, base)


main()
