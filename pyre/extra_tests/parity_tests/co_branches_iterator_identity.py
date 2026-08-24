# CPython-suite gap: `test_monitoring` and `test_code` read the *values*
# `co_branches()` yields and never look at the object producing them, so a
# runtime that materializes the whole list up front and hands back a generic
# sequence iterator passes both modules.
#
# parity-tests reason: `co_branches()` returns a `branchesiterator`, whose
# `tp_name` is `line_iterator` — the same string `_PyLineIterator` carries, so
# the two distinct types are told apart only by identity.  The walk is lazy:
# `next()` resumes the bytecode scan at `bi_offset` and returns at the first
# branch it reaches, so reading one row does not cost a pass over the whole
# code object.  Both iterators are `Py_TPFLAGS_BASETYPE` with no `tp_new`.

def branchy(x):
    for i in range(3):
        if i:
            x += i
        elif x:
            x -= 1
    while x > 0:
        x -= 1
    return x


CODE = branchy.__code__
it = CODE.co_branches()

print("type name:", type(it).__name__)
print("distinct from co_lines:", type(it) is not type(CODE.co_lines()))
print("iter is self:", iter(it) is it)
print("rows:", list(it))
print("exhausted:", list(it))

# Laziness is observable: an iterator only stepped once has not walked past the
# branch it reported, so a second iterator over the same code starts over and
# agrees row for row.
first = CODE.co_branches()
print("first row:", next(first))
print("full walk agrees:", list(CODE.co_branches())[0] == next(iter(CODE.co_branches())))

# No `tp_new`, but `Py_TPFLAGS_BASETYPE` is set.
try:
    type(it)()
except TypeError as exc:
    print("ctor:", exc)


class Sub(type(it)):
    pass


print("subclassable:", Sub.__mro__[1] is type(it))

# The unbound `__next__` of the *other* type named `line_iterator` rejects this
# receiver, and reports both types by that shared name.
try:
    type(CODE.co_lines()).__next__(CODE.co_branches())
except TypeError as exc:
    print("cross receiver:", exc)

# An `async for` contributes its END_ASYNC_FOR row.
SRC = """
async def drain(aiterable):
    async for item in aiterable:
        pass
"""
ns = {}
exec(compile(SRC, "<branches>", "exec"), ns)
print("async rows:", list(ns["drain"].__code__.co_branches()))

# A code object with no branch at all yields nothing.
print("no branches:", list((lambda: 1).__code__.co_branches()))

print("OK")
