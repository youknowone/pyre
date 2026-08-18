# CPython-suite gap: `test_funcattrs` reassigns `__code__` and then calls the
# function once, never from inside a loop hot enough to have inlined the old
# body.
# parity-tests reason: this is a pyre JIT inline-lever regression.

# `function.py:47 _immutable_fields_ = ['code?', 'w_func_globals?',
# 'closure?[*]', 'defs_w?[*]']`.  The `?` is what lets the inline lever bake
# `code` and equally what registers the invalidation an assignment to the slot
# owes.  The lever bakes it in the strongest form there is — `code` selects
# which callee body the trace walks into — so without the `?` a loop keeps
# running a body the function no longer has.
#
# The per-iteration `getfield_gc_r` + `guard_value` the lever emits elsewhere
# cannot stand in here: it reads the field off the pinned operand, and for a
# constant callable that operand is a baked `ConstPtr`.
#
# Each reassignment happens INSIDE its loop.  A call after the loop is
# interpreted and would not consult what the trace baked.

N = 40000
SWITCH = N // 2


def small():
    return 1


def big():
    return 500


def module_level_callee():
    # The constant-callable shape: `small` is resolved once and baked.
    total = 0
    i = 0
    while i < N:
        total += small()
        if i == SWITCH:
            small.__code__ = big.__code__
        i += 1
    expected = (SWITCH + 1) + (N - SWITCH - 1) * 500
    assert total == expected, 'baked a stale __code__: %r != %r' % (total, expected)


class Holder:
    def m(self):
        return 1


def m_big(self):
    return 500


def method_callee():
    # The method shape: the receiver's type version pins the descriptor, which
    # says nothing about the function's own `code` slot.
    obj = Holder()
    total = 0
    i = 0
    while i < N:
        total += obj.m()
        if i == SWITCH:
            Holder.m.__code__ = m_big.__code__
        i += 1
    expected = (SWITCH + 1) + (N - SWITCH - 1) * 500
    assert total == expected, 'baked a stale method __code__: %r != %r' % (total, expected)


def getter_small(self):
    return 1


def getter_big(self):
    return 500


class WithProperty:
    x = property(getter_small)


def property_accessor_callee():
    # The property fold resolves the accessor to a trace constant, so it lands
    # on the same arm the module-level callee does.
    obj = WithProperty()
    total = 0
    i = 0
    while i < N:
        total += obj.x
        if i == SWITCH:
            getter_small.__code__ = getter_big.__code__
        i += 1
    expected = (SWITCH + 1) + (N - SWITCH - 1) * 500
    assert total == expected, 'baked a stale accessor __code__: %r != %r' % (total, expected)


def hook_small(self, name):
    return 1


def hook_big(self, name):
    return 500


class WithHook:
    pass


WithHook.__getattr__ = hook_small


def getattr_hook_callee():
    # The `__getattr__` fold resolves its callee the same way.
    obj = WithHook()
    total = 0
    i = 0
    while i < N:
        total += obj.absent
        if i == SWITCH:
            hook_small.__code__ = hook_big.__code__
        i += 1
    expected = (SWITCH + 1) + (N - SWITCH - 1) * 500
    assert total == expected, 'baked a stale hook __code__: %r != %r' % (total, expected)


def fresh_callee_still_inlines():
    # A `MAKE_FUNCTION` in the loop body allocates a fresh callee every
    # iteration, so the lever keeps re-proving `code` off the live function
    # instead.  Here only to catch that arm being given up along the way.
    total = 0
    i = 0
    while i < N:
        def helper(x):
            return x + 1
        total += helper(i)
        i += 1
    assert total == N * (N + 1) // 2, 'fresh-callee inline changed answer: %r' % (total,)


module_level_callee()
method_callee()
property_accessor_callee()
getattr_hook_callee()
fresh_callee_still_inlines()
print("OK")
