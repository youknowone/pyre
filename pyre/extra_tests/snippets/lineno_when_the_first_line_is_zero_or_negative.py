# pyre-check: gate=1
# pyre-check: pypy-diverges: rebuilds a code object through `types.CodeType` to reach `co_firstlineno <= 0`; pypy3 has no `co_exceptiontable` to hand it
# CPython-suite gap: `test_code`, `test_frame` and `test_traceback` only ever
# build code objects whose `co_firstlineno` is a real source line, and
# `code.replace(co_firstlineno=...)` rejects anything below 1.  The
# `types.CodeType` constructor does not, and `marshal` round-trips whatever it
# was given, so a runtime that stores the first line as a one-indexed value can
# pass every module while decoding the whole line table against a different
# origin than the one the code object reports.
#
# parity-tests reason: `_PyCode_InitAddressRange` seeds the walk with
# `co->co_firstlineno` whatever its sign, and every line the table computes is
# that origin plus a delta.  Both readers of a *resolved* line -- `tb_lineno`
# and `f_lineno` -- answer `None` below zero, while `repr(frame)` prints the
# number itself, so the three disagree on purpose and only a signed resolver
# gets all three right.  `co_lines()` and `co_positions()` add a third rule:
# `-1` exactly is the missing-value marker and reports as `None` there, so a
# line that *computes* to `-1` is indistinguishable from a `NO_LOCATION` range.
#
# PyPy 7.3.20 is a 3.11 line table with no `co_exceptiontable`, so the
# `types.CodeType` call below raises there before any of this is reached.

import re
import sys
import types

SOURCE = "def probe(box):\n    box.append(sys._getframe())\n    return 1\n"
NAMESPACE = {"sys": sys}
exec(compile(SOURCE, "<origin>", "exec"), NAMESPACE)
BASE = NAMESPACE["probe"].__code__


def with_first_line(firstlineno):
    return types.CodeType(
        BASE.co_argcount,
        BASE.co_posonlyargcount,
        BASE.co_kwonlyargcount,
        BASE.co_nlocals,
        BASE.co_stacksize,
        BASE.co_flags,
        BASE.co_code,
        BASE.co_consts,
        BASE.co_names,
        BASE.co_varnames,
        BASE.co_filename,
        BASE.co_name,
        BASE.co_qualname,
        firstlineno,
        BASE.co_linetable,
        BASE.co_exceptiontable,
        BASE.co_freevars,
        BASE.co_cellvars,
    )


for first in (-8, -1, 0, 1):
    code = with_first_line(first)
    box = []
    types.FunctionType(code, NAMESPACE)(box)
    frame = box[0]
    print("co_firstlineno:", code.co_firstlineno)
    # The first range covers the `RESUME`, which carries the origin itself.
    print("   co_lines:", list(code.co_lines())[:2])
    print("   co_positions:", list(code.co_positions())[:2])
    # `frame_getlineno` hides a negative line; `frame_repr` prints it.
    print("   f_lineno:", frame.f_lineno)
    print("   repr:", re.sub(r"0x[0-9a-f]+", "0xADDR", repr(frame)))
    # `tb_lineno` resolves the stored `-1` sentinel against the same origin.
    resolved = [
        types.TracebackType(None, frame, lasti, -1).tb_lineno
        for lasti in (-2, -1, 0, 2, 4, 10000)
    ]
    print("   tb_lineno:", resolved)

print("OK")
