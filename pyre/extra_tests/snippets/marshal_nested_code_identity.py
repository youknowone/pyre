# `test_marshal` is not in the CPython-suite baseline, so nothing in the suite
# loads a stream in which one code object appears twice.
#
# `unmarshal_pycode` is declared with `save_ref=True` and calls `save_ref` on
# the code object it is about to fill in, so a second occurrence of that object
# travels as a back-reference. The loader owes both spellings the same object.
#
# Reaching that requires the decoded wrapper to be what the parent's constant
# slot keeps. A loader that instead rebuilds the slot from its own copy of the
# body answers with an equal-but-distinct code object, which is invisible to
# every test that compares constants by value.
import marshal


def f():
    def g():
        return 1

    return g


code = f.__code__
nested = [c for c in code.co_consts if hasattr(c, "co_name")][0]

# `nested` occurs twice in the stream: once inside `code.co_consts`, and once as
# the tuple's own second item, which is written as a reference to the first.
loaded_code, loaded_nested = marshal.loads(marshal.dumps((code, nested)))
from_consts = [c for c in loaded_code.co_consts if hasattr(c, "co_name")][0]

assert from_consts is loaded_nested, "nested code object identity was not preserved"

# The retained wrapper is a working code object, not just the right identity.
assert from_consts.co_name == "g"
assert eval(marshal.loads(marshal.dumps(compile("21 * 2", "<pin>", "eval")))) == 42
