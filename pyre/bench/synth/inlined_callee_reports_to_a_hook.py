# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=settrace_direct,settrace_nested,setprofile_direct,setprofile_nested,root:callee,root:wrapper
# The two `root:` arms are the premise, not a relaxation: the events this fixture
# counts are owed precisely because compiled code enters the callee past
# `execute_frame`'s bracket.  A callee that stopped reaching the JIT would make
# every count below pass without testing anything.
# Self-checking guard for the events a callee owes a hook when the loop that
# calls it is running compiled.
#
# `pyframe.py PyFrame.execute_frame` brackets every Python call with
# `ec.call_trace(self)` and `ec.return_trace(self, w_exitvalue)`, and
# `executioncontext.py ExecutionContext.leave` adds `_trace(frame,
# 'leaveframe', ...)` when a profiler is installed.  A tracing JIT reaches a
# callee through doors that begin at the callee's first bytecode -- an inlined
# body, or a direct jump into the callee's own compiled trace -- and every one
# of them starts past that bracket.  Upstream keeps the events anyway because
# it traces THROUGH `execute_frame`: the `gettrace()` reads become guards in
# the compiled loop, so a hook installed later fails one.  An implementation
# that synthesizes the callee entry instead has to answer for them some other
# way, or the loop runs to completion with the hook installed and silent.
#
# THE TAIL IS THE TEST, AND SO IS RUNNING IT TWICE.  The loss is total rather
# than partial: the hook is installed mid-loop, and what arrives is the handful
# of events from the re-warmup before compiled code takes over again -- a
# number set by the JIT's own threshold, NOT by how long the tail is.  So each
# column is asserted at two tails, and a count that reads the same at both is
# reported as the blind-forever signature it is.  Measured before the fix:
# 1043 at a 2500 tail and 1043 at a 10000 one, against exactly the tail on
# cpython 3.14.6 and on pypy3 7.3.22.
#
# The `nested` column is the control and it is on the other side of the line:
# the loop calls a wrapper, and the counted callee is invoked from THAT frame,
# which is interpreted.  It reports on every implementation, which is what
# makes a short `direct` column attributable to the door compiled code enters a
# callee through and not to the setter, the tail length or the counting rule.
#
# `sys.settrace` and `sys.setprofile` are both exercised because they are
# answered by different machinery: a profiler is part of the portal driver's
# green key (`interp_jit.py greens = ['next_instr', 'is_being_profiled',
# 'pycode']`) so arming one mints a different cell, while a trace function has
# no green and needs the compiled loop to be carrying a guard on the slot
# `gettrace()` reads.  An implementation that answers one and not the other
# passes exactly one pair of columns here.
import sys

WARM = 20000  # past the JIT's compilation threshold many times over
TAILS = (2500, 10000)

calls = 0


def hook(frame, event, arg):
    global calls
    if event == 'call' and frame.f_code.co_name == 'callee':
        calls += 1
    return hook


def callee(x):
    return x + 1


def wrapper(x):
    return callee(x)


# One loop per column, spelled out rather than generated.  Two reasons, both
# load-bearing.  A column that reused another column's loop would inherit
# whatever state the previous arm left on it -- a green key already
# deoptimized, a code object already marked don't-trace-here -- and could then
# report every event for the plainest reason of all: never having compiled; a
# fresh name is a fresh green key (`interp_jit.py greens` carries `pycode`).
# And a loop built by `exec` into a fresh dict is NOT the same subject: its
# globals are a plain dict rather than a module's, which is its own JIT path,
# and measured, every column built that way reads 2 -- including the control
# that must pass.  Four near-identical bodies is the price of four columns
# that are each the shape this file is about.


def settrace_direct(n, arm_at):
    acc = 0
    i = 0
    while i < n:
        if i == arm_at:
            sys.settrace(hook)
        acc = callee(acc)
        i += 1
    return acc


def settrace_nested(n, arm_at):
    acc = 0
    i = 0
    while i < n:
        if i == arm_at:
            sys.settrace(hook)
        acc = wrapper(acc)
        i += 1
    return acc


def setprofile_direct(n, arm_at):
    acc = 0
    i = 0
    while i < n:
        if i == arm_at:
            sys.setprofile(hook)
        acc = callee(acc)
        i += 1
    return acc


def setprofile_nested(n, arm_at):
    acc = 0
    i = 0
    while i < n:
        if i == arm_at:
            sys.setprofile(hook)
        acc = wrapper(acc)
        i += 1
    return acc


def run(loop, unsetter, tail):
    global calls
    calls = 0
    loop(WARM + tail, WARM)
    unsetter(None)
    return calls


def main():
    columns = [('settrace/direct', settrace_direct, sys.settrace),
               ('settrace/nested', settrace_nested, sys.settrace)]
    setprofile = getattr(sys, 'setprofile', None)
    if setprofile is not None:
        columns.append(('setprofile/direct', setprofile_direct, setprofile))
        columns.append(('setprofile/nested', setprofile_nested, setprofile))

    failures = []
    for name, loop, unsetter in columns:
        seen = [run(loop, unsetter, tail) for tail in TAILS]
        if seen == list(TAILS):
            continue
        failures.append('%s: got %r for tails %r' % (name, seen, list(TAILS)))
        if seen[0] == seen[1]:
            failures.append(
                '%s: the count does not track the tail, so the loop is blind '
                'from the arming onward' % (name,)
            )
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a callee called from a compiled loop reports to a hook')
    return 0


sys.exit(main())
