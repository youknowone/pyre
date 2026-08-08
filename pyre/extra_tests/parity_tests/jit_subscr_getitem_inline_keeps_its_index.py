"""A user `__getitem__` inlined from BINARY_OP keeps its own index operand.

The FOR_ITER deferred-inline gate used to admit this inline because it carried
no `arg_class_guard`, which stood for "entered from a CALL the abort rewind can
name". `obj[key]` enters from BINARY_OP instead, so a deferred abort resumed one
operand short and the subscript index was replaced at runtime by an unrelated
live Ref — here the enclosing generator's iterator, which surfaced as

    TypeError: list indices must be integers or slices, not list_iterator

from `re._parser.SubPattern.__getitem__`. JIT-only; `PYRE_NO_JIT=1` always
passed.
"""

import re

pattern = re.compile("|".join("%d" % x for x in range(10000)))
assert pattern.match("9999") is not None
assert pattern.match("beef") is None

print("OK")
