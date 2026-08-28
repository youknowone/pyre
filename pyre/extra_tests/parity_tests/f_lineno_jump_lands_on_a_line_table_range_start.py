# CPython-suite gap: `test_sys_settrace`'s jump tests all move between lines a
# statement actually starts on, inside a function whose first instruction
# already carries `co_firstlineno`.  None of them jumps to the first line of a
# *module* body, whose `RESUME` sits in a range carrying line 0, and none of
# them jumps out of the body of a `with`, so a runtime can pass every module
# while resolving jump targets off an expanded per-instruction table and while
# assuming every stack slot it unwinds is bound.
#
# parity-tests reason: `marklines` marks the **start** unit of each line-table
# range, and skips a range whose line is `-1`.  A table expanded into one entry
# per instruction can express neither -- its line is a one-indexed integer, so
# the module `RESUME`'s line 0 reads back as 1 and the `RESUME` itself becomes
# a legal target for `f_lineno = 1`.  The unwind is the other half: upstream
# closes each dropped slot with `PyStackRef_XCLOSE`, and a jump out of a `with`
# body drops exactly one slot that is still unbound, so a pop that requires a
# value aborts the interpreter rather than raising.
#
# PyPy 7.3.20 reports the module `RESUME` differently -- it is a 3.11 line
# table and has no line-0 range there -- so it fails the first case for the
# same reason `traceback_lineno_sentinel.py` gives.

import sys

MODULE_SOURCE = "x = 1\ny = 2\n"
MODULE_JUMP = []


def module_tracer(frame, event, arg):
    if frame.f_code.co_filename != "<jump>":
        return None
    if event == "call":
        frame.f_trace = module_tracer
        frame.f_trace_lines = True
        return module_tracer
    if event == "line" and not MODULE_JUMP:
        try:
            frame.f_lineno = 1
        except BaseException as exc:
            MODULE_JUMP.append((type(exc).__name__, str(exc)))
        else:
            MODULE_JUMP.append((frame.f_lasti, frame.f_lineno))
    return module_tracer


sys.settrace(module_tracer)
try:
    exec(compile(MODULE_SOURCE, "<jump>", "exec"), {})
finally:
    sys.settrace(None)

# The `RESUME` is a range start carrying line 0, so it is not where line 1
# begins; line 1 begins at the next instruction unit, byte offset 2.
print("module jump to line 1:", MODULE_JUMP)


WITH_SOURCE = """
def run(cm, log):
    with cm:
        log.append('body')
    log.append('after')
    return 'done'
"""

NAMESPACE = {}
exec(compile(WITH_SOURCE, "<with>", "exec"), NAMESPACE)


class Manager:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        LOG.append("exit")
        return False


LOG = []
WITH_JUMP = []


def with_tracer(frame, event, arg):
    if frame.f_code.co_filename != "<with>":
        return None
    if event == "call":
        frame.f_trace = with_tracer
        frame.f_trace_lines = True
        return with_tracer
    if event == "line" and frame.f_lineno == 4 and not WITH_JUMP:
        WITH_JUMP.append("attempted")
        try:
            frame.f_lineno = 5
        except BaseException as exc:
            WITH_JUMP.append((type(exc).__name__, str(exc)))
        else:
            WITH_JUMP.append(("landed", frame.f_lineno))
    return with_tracer


sys.settrace(with_tracer)
try:
    RESULT = NAMESPACE["run"](Manager(), LOG)
finally:
    sys.settrace(None)

# Leaving the block this way abandons it: the bound `__exit__` is dropped off
# the stack unbound rather than called, so nothing appends 'exit'.
print("with jump to line 5:", WITH_JUMP)
print("with jump log:", LOG, "result:", RESULT)

print("OK")
