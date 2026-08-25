# CPython-suite gap: `test_sys_settrace` compares full event lists, but every
# expectation it carries was written from a function body, and `test_pdb`'s
# doctests that would catch this live in `test_zipimport_support` -- gated, and
# only reached by running a zipped copy of the doctest module.
#
# parity-tests reason: `initialize_lines` refuses to instrument five opcodes as
# line-event sites -- `RESUME`, `POP_ITER`, `END_FOR`, `END_SEND` and
# `END_ASYNC_FOR` -- so `INSTRUMENTED_LINE` is never written over them and no
# `line` event can fire there.  The prologue `RESUME` is the one every frame
# has, and a module body's carries line 0, below `co_firstlineno`: a runtime
# without the filter reports an extra `line` event at line 0 before the first
# statement, and `pdb` stops on `<string>(0)` where 3.14 stops on `(1)`.
#
# The `call` event's `f_lasti` is deliberately not compared: `_PyInterpreterFrame
# _LASTI` is 0 while the frame sits on its `RESUME`, where this runtime reports
# "not started", and that convention gap is a separate open item.
#
# PyPy 7.3.20 is a 3.11 line table whose module body has no line-0 range, so
# its event list differs for that reason.

import sys

MODULE_EVENTS = []
FUNCTION_EVENTS = []


def module_tracer(frame, event, arg):
    if frame.f_code.co_filename != "<events>":
        return None
    MODULE_EVENTS.append((event, frame.f_lineno))
    return module_tracer


CODE = compile("x = 12\ny = x\n", "<events>", "exec")
print("co_lines:", list(CODE.co_lines()))
sys.settrace(module_tracer)
try:
    exec(CODE, {})
finally:
    sys.settrace(None)
print("module events:", MODULE_EVENTS)


SOURCE = """
def target(a):
    b = a + 1
    for i in range(2):
        b += i
    return b
"""
NAMESPACE = {}
exec(compile(SOURCE, "<events>", "exec"), NAMESPACE)


def function_tracer(frame, event, arg):
    if frame.f_code.co_name != "target":
        return None
    FUNCTION_EVENTS.append((event, frame.f_lineno))
    return function_tracer


sys.settrace(function_tracer)
try:
    NAMESPACE["target"](1)
finally:
    sys.settrace(None)
# The loop's `END_FOR` / `POP_ITER` are excluded for the same reason, so the
# exit carries no event of its own.
print("function events:", FUNCTION_EVENTS)

print("OK")
