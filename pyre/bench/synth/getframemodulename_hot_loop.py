# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=root:_callee_template
# The `root:` arm is measured, not a relaxation: this fixture's loop aborts
# five times with ABORT_ESCAPE and what reaches the JIT is the root trace
# `finish_and_compile` attaches. The cause is that no fold row exists for
# `sys._getframemodulename`: the fold's head gate tests
# `is_builtin_getframe_function(concrete_callable)`, a fn-pointer whitelist a
# separate closure cannot match. Substituting `sys._getframe(0)` at the same
# site compiles, and suppressing the getframe fold on that substitute
# reproduces this fixture byte for byte. Folding the depth-0 read alone is
# measured sufficient. ⚠ Because `hot` never compiles a loop, the depth-0
# assertion below is checked on interpreted iterations only.
# Self-checking regression guard for `sys._getframemodulename` read from a hot
# loop, at depth 0 and from one frame further in.
#
# It is the third forcing reader of a calling frame in the `sys` module, beside
# `getframe` and `_current_frames`, and it had no coverage anywhere -- neither
# a synthetic fixture nor a snippet.  It cannot join `current_frames_hot_loop`,
# which is an ordinary synthetic fixture and therefore also run under PyPy for
# the ratio: PyPy does not implement `_getframemodulename`, and the run dies
# with AttributeError before it reaches any assertion.  A self-checking fixture
# runs on pyre alone, which is what makes this surface reachable at all.
#
# The answer is invariant across the loop, so it is collected as a SET rather
# than compared per iteration: a route that went stale, or that started
# answering per-iteration once the loop compiled, shows up as a second element.
# A scalar comparison against the expected value would accept a value that
# oscillated and happened to be right on the last iteration.
#
# Depth 1 is read from a callee the tracer can inline, so the answer has to
# name the CALLER's module rather than the frame the walk is standing in.  The
# callee therefore runs under a module of its OWN: with caller and callee both
# in `__main__` the two answers coincide, and an implementation that ignored
# the requested depth entirely -- always reporting the frame it is standing in
# -- produced the expected `{"__main__"}` and passed.  Measured on the shipped
# shape: depth 1 and depth 0 both answered `__main__`.  With the callee moved
# out, depth 0 answers `OTHER_MODULE` and depth 1 still answers `__main__`, so
# the two are finally distinguishable -- and BOTH are asserted, because pinning
# only depth 1 leaves a route that has stopped distinguishing them unremarked.
#
# The callee is built by rebinding a code object against the other module's
# namespace rather than by `exec`-ing source into it: `exec` into a namespace
# that is not a module's own dict is a separate open defect here, and this
# fixture has no reason to depend on it.
import sys

N = 200000
EXPECTED = "__main__"
OTHER_MODULE = "pyre_getframemodulename_callee"

_other = type(sys)(OTHER_MODULE)
_other.sys = sys


def _callee_template(seen_depth1):
    seen_depth1.add(sys._getframemodulename(1))


def _callee_depth0_template():
    return sys._getframemodulename(0)


inner_depth1 = type(_callee_template)(
    _callee_template.__code__, _other.__dict__, "inner_depth1"
)
callee_depth0 = type(_callee_depth0_template)(
    _callee_depth0_template.__code__, _other.__dict__, "callee_depth0"
)


def hot(n):
    at0 = set()
    at1 = set()
    for _ in range(n):
        at0.add(sys._getframemodulename(0))
        inner_depth1(at1)
    return at0, at1


def main():
    at0, at1 = hot(N)
    # Read once, OUTSIDE the hot loop.  Every `_getframemodulename` forces the
    # frame -- it reads `w_globals`, a redirected field -- so a third call in
    # the loop body costs a third escape per iteration and the loop stops
    # compiling altogether (measured: `loops_compiled` 1 -> 0, `abrt_escape`
    # 5 -> 10).  The depth-0 answer is invariant, so one read establishes it
    # and the hot loop keeps guarding what it exists to guard: the depth-1
    # answer from a callee the tracer inlines, in COMPILED code.
    callee_at0 = {callee_depth0()}
    if at0 != {EXPECTED}:
        print("FAIL _getframemodulename(0) not invariant:", sorted(at0, key=str))
        return 1
    if at1 != {EXPECTED}:
        print("FAIL _getframemodulename(1) from an inlinable callee:",
              sorted(at1, key=str))
        return 1
    # The callee's own depth 0.  Distinct from `at1` by construction, so a
    # route that collapsed the two -- ignoring the requested depth -- fails
    # here even while `at1` still reads `__main__`.
    if callee_at0 != {OTHER_MODULE}:
        print("FAIL _getframemodulename(0) inside the foreign-module callee:",
              sorted(callee_at0, key=str))
        return 1
    print("PASS _getframemodulename hot loop")
    return 0


sys.exit(main())
