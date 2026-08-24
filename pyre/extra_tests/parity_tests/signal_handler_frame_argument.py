# pyre-check: platforms=linux,darwin
# CPython-suite gap: `test_signal` declares `frame` in the signature of every
# handler it defines and then asserts nothing about it -- no test in the module
# reads `f_code`, `f_locals` or `f_back` off the argument -- so a runtime that
# hands every handler `None` passes the whole file.
#
# parity-tests reason: the second argument is the interpreter's own call stack
# handed to application code, which is the point a JIT owes a writeback of the
# fields it was keeping out of the frame. Handing over `None` instead is
# invisible to the suite and takes every `f_back` walk with it, which is how a
# signal-driven debugger or stack dumper finds out where it interrupted.
import os
import signal

seen = []


def handler(signum, frame):
    seen.append(frame)


def target():
    marker = 'target-local'
    os.kill(os.getpid(), signal.SIGUSR1)
    # Delivery happens at a bytecode boundary rather than inside `kill`, so
    # the handler has not run yet on the line above.
    for _ in range(1000):
        if seen:
            break
    return marker


signal.signal(signal.SIGUSR1, handler)
assert target() == 'target-local'
assert len(seen) == 1, seen

frame = seen[0]
assert frame is not None, 'the handler was handed no frame'
# The frame is the one that was interrupted, not the handler's own.
assert frame.f_code.co_name == 'target', frame.f_code.co_name
# Its locals are readable, which is what a frame the JIT was running owes
# before application code reads it.
assert frame.f_locals['marker'] == 'target-local', frame.f_locals
# The walk a debugger makes from there.
assert frame.f_back is not None, 'the interrupted frame has no caller'
assert frame.f_back.f_code.co_name == '<module>', frame.f_back.f_code.co_name
assert isinstance(frame.f_lineno, int), frame.f_lineno

print('OK')
