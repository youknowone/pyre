# CPython-suite gap: no suite test resumes a JIT frame at a LOAD_ATTR whose
# inlined descriptor body aborted.
# parity-tests reason: this is a pyre trace-abort operand-stack regression.

# The operand stack an aborted inline sub-walk hands back to the interpreter,
# for an inline entered from LOAD_ATTR rather than from CALL.
#
# A `__getattr__` hook and a `property` getter are both inlined in place of the
# attribute residual, so both enter the inline lever from LOAD_ATTR.  When the
# sub-walk gives up, the caller's frame is flushed at that opcode and the
# interpreter re-executes it, which means the flush has to rebuild the operand
# stack the LOAD_ATTR pops.  One of the sources it rebuilds from is the encoded
# residual's Ref operand list.  For a CALL that list is exactly the stack image
# (`[callable, null_or_self, args...]`); for `load_attr_fn(obj, code, name_idx)`
# it is `[obj, code]`, whose `code` is a code object the Python stack never
# held.  Publishing it resumed the LOAD_ATTR with the code object as receiver:
# `AttributeError: 'code' object has no attribute 'missing'` for an attribute
# the hook answers.
#
# Reaching the abort needs all three of: a FOR_ITER caller (the same body under
# `while` is admitted through an arm that does not abort), a body admitted as
# deferred-call safe, and a residual inside that body which does not inline —
# the string concatenations below.

N = 3000


class HookOwner:
    def __getattr__(self, name):
        return "hook:" + name


class PropertyOwner:
    def __init__(self):
        self._value = "v"

    @property
    def value(self):
        return "prop:" + self._value


def read_hook(owner):
    last = None
    for _ in range(N):
        last = owner.missing
    return last


def read_property(owner):
    last = None
    for _ in range(N):
        last = owner.value
    return last


assert read_hook(HookOwner()) == "hook:missing"
assert read_property(PropertyOwner()) == "prop:v"

print("OK")
