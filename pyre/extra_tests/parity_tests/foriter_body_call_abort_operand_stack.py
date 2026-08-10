# CPython-suite gap: iteration tests cannot exercise a JIT inline-walk abort stack.
# parity-tests reason: this is a pyre trace-abort operand-stack regression.

# The operand stack an aborted inline sub-walk hands back to the interpreter,
# for a call made inside a FOR_ITER body.
#
# When the walker gives up on inlining a callee it flushes the caller's frame
# at the CALL it entered under and lets the interpreter re-execute the whole
# call.  That flush rebuilds the operand stack from two sources: the enclosing
# FOR_ITER's iterator (and anything else below the call) from the vstack
# mirror, and the call's own operands from the encoded residual op.  The
# operands do not sit at a fixed byte offset in that op — the method-form CALL
# helpers lower through a shape whose leading Int list is variable-width, so a
# reader that assumes the plain shape's offset picks up the Int list's register
# indices and resolves them in the Ref bank.  The result is a stack of the
# right HEIGHT (so the flush's depth check passes) holding the wrong objects:
# here the enclosing loop's iterator arrived as the subscript index, and
# `SubPattern.__getitem__` raised `TypeError: list indices must be integers or
# slices, not list_iterator`.
#
# `re.compile` of a large flat alternation is the reproducer: `_compile_info`
# calls `_get_charset_prefix`, whose BRANCH arm loops `for p in av[1]` and
# subscripts `p[0]` inside the body.  The alternation has to be big enough for
# that loop to go hot — it is clean below about a thousand branches — and no
# hand-written class with the same shape has been made to reach the abort leg,
# so the real module drives it.
#
# The compile is the assertion: any wrong operand raises out of `re`.

import re

BRANCHES = 2000

pattern = "|".join("%d" % x for x in range(BRANCHES))
compiled = re.compile(pattern)

# Alternation is first-match, so "1999" is matched by the earlier "1" branch.
assert compiled.match("1999").group(0) == "1"
assert compiled.match("nope") is None

print("OK")
