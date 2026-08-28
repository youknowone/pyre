# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=driver,root:leaf
# Both entries are the premise, not a relaxation. The caller whose vref goes
# unforced is `mid_pass`, and it is only virtual because `driver`'s loop
# compiles and inlines it; `leaf` is the residual callee whose own activation
# reaches `leave_compiled_frame_chain`, which is the port under test. A run
# where either stopped compiling would leave `mid_pass` a real interpreter
# frame with no vref at all, and every assertion below would pass without
# testing anything.
#
# `executioncontext.py ExecutionContext.leave` reaches the caller WITH the
# force:
#
#     if frame.escaped or got_exception:
#         f_back = frame.f_backref()      # parens == force_virtual
#         if f_back:
#             f_back.mark_as_escaped()
#         frame_vref()
#
# pyre's port for a frame whose body ran as compiled code
# (`call_jit.rs leave_compiled_frame_chain`) read that caller through the
# NON-forcing `vref_referent` instead. For a caller the JIT kept virtual that
# answers NULL, NULL was taken for "no caller", and the caller was neither
# materialised nor marked escaped.
#
# The mark is not bookkeeping: it is what makes the caller's own leave record
# the `VIRTUAL_REF_FINISH(vrefbox, virtualbox)` form, the one that stores the
# virtual into `forced`. Without the mark, `walker_ec_leave` takes its
# unescaped arm and records the NULL form, which leaves the caller's vref at
# `virtual_token == TOKEN_NONE` with `forced == NULL` — the state
# `virtualref.py` calls `InvalidVirtualRef`. `force_pyframe_vref` says so
# itself: the state that reaches its refusal "requires that propagation to have
# been skipped".
#
# THE SHAPE IS THE TEST, and three parts of it are load-bearing:
#   * `leaf` must be a RESIDUAL call, so its frame reaches
#     `leave_compiled_frame_chain` rather than the walker's own leave.
#   * `leaf` must ESCAPE — its frame is retained out of a traceback — so the
#     escape branch runs at all.
#   * `leaf` must return NORMALLY. An exception leaving `leaf` fails the
#     trace's `GuardNoException`, which skips the NULL-form finish and installs
#     a tracing-time vref that already carries `forced`; the bug goes quiet.
#
# Before the fix this did not print a wrong answer — it ABORTED the process:
# `panicked at pyre-jit/src/eval.rs: InvalidVirtualRef: frame-chain vref forced
# after its frame died` -> `thread caused non-unwinding panic. aborting.`,
# exit 134, while cpython 3.14, pypy3 and `PYRE_JIT=0` all answered `mid_pass`.
# An abort is why this fixture asserts rather than prints: there is no output to
# diff against when the defect fires.
import sys

WARM = 4000
ESCAPE_AT = WARM - 5


def leaf(i, n):
    if i == ESCAPE_AT:
        try:
            raise ValueError('escape')
        except ValueError as exc:
            HELD.append(exc.__traceback__.tb_frame)
    return i


def mid_pass(i, n):
    return leaf(i, n)


def driver(n):
    acc = 0
    i = 0
    while i < n:
        acc += mid_pass(i, n)
        i += 1
    return acc


HELD = []


def main():
    failures = []
    driver(WARM)
    if len(HELD) != 1:
        print('FAIL the escape never fired: held %d frames' % len(HELD))
        return 1
    escaped = HELD[0]
    if escaped.f_code.co_name != 'leaf':
        failures.append('escaped frame = %s, expected leaf' % escaped.f_code.co_name)
    # The read that aborts. `f_back` forces the caller's vref, which is exactly
    # the vref the missing propagation stranded.
    caller = escaped.f_back
    if caller is None:
        failures.append('f_back = None, expected the mid_pass frame')
    elif caller.f_code.co_name != 'mid_pass':
        failures.append('f_back = %s, expected mid_pass' % caller.f_code.co_name)
    else:
        # One level further: the caller's own caller must be reachable too, so
        # the propagation is checked past the frame that was skipped rather
        # than only at it.
        grandparent = caller.f_back
        name = None if grandparent is None else grandparent.f_code.co_name
        if name != 'driver':
            failures.append('f_back.f_back = %s, expected driver' % name)
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS the portal leave forces the caller vref it marks escaped')
    return 0


sys.exit(main())
