# CPython-suite gap: no suite test rebinds `__next__`, `__iter__`, `__index__`,
# `__getattr__` or `__getattribute__` twice and then keeps using the slot in a
# loop hot enough to compile.  The second rebind is the one that matters.
#
# `typeobject.py W_TypeObject.setdictvalue` routes every class-namespace store
# through `write_cell`.  The store that replaces the class-body value builds an
# `ObjectMutableCell` and calls `mutated()`.  Every store after that writes
# `w_value` inside the installed cell, `write_cell` returns `None`, and
# `mutated()` is skipped -- so no version tag moves.
#
# parity-tests reason: a trace that answers the slot under a version-tag pin
# alone is correct until the first rebind and wrong from the second on.  Each
# case below parks the cell, then rebinds INSIDE the loop so the compiled code
# has to read `ObjectMutableCell.w_value`.
#
# The store sits after the read, so iterations `0..=SWITCH` read the first
# value and the rest read the second.
N = 20000
SWITCH = N // 2


def expect(total, first, second, what):
    wanted = (SWITCH + 1) * first + (N - SWITCH - 1) * second
    assert total == wanted, '%s: %r != %r' % (what, total, wanted)


class It:
    def __next__(self):
        return 1


It.__next__ = lambda self: 1


def user_next():
    obj = It()
    total = 0
    i = 0
    while i < N:
        total += next(obj)
        if i == SWITCH:
            It.__next__ = lambda self: 7
        i += 1
    expect(total, 1, 7, 'stale __next__')


class Iterable:
    def __iter__(self):
        return iter((1,))


Iterable.__iter__ = lambda self: iter((1,))


def user_iter():
    obj = Iterable()
    total = 0
    i = 0
    while i < N:
        total += next(iter(obj))
        if i == SWITCH:
            Iterable.__iter__ = lambda self: iter((7,))
        i += 1
    expect(total, 1, 7, 'stale __iter__')


class Idx:
    def __index__(self):
        return 1


Idx.__index__ = lambda self: 1


def user_index():
    obj = Idx()
    lst = [0, 10, 0, 0, 0, 0, 0, 70]
    total = 0
    i = 0
    while i < N:
        total += lst[obj]
        if i == SWITCH:
            Idx.__index__ = lambda self: 7
        i += 1
    expect(total, 10, 70, 'stale __index__')


class Ga:
    def __getattr__(self, name):
        return 1


Ga.__getattr__ = lambda self, name: 1


def user_getattr():
    obj = Ga()
    total = 0
    i = 0
    while i < N:
        total += obj.missing
        if i == SWITCH:
            Ga.__getattr__ = lambda self, name: 7
        i += 1
    expect(total, 1, 7, 'stale __getattr__')


class Gattr:
    def __getattribute__(self, name):
        return 1


Gattr.__getattribute__ = lambda self, name: 1


def user_getattribute():
    obj = Gattr()
    total = 0
    i = 0
    while i < N:
        total += obj.x
        if i == SWITCH:
            Gattr.__getattribute__ = lambda self, name: 7
        i += 1
    expect(total, 1, 7, 'stale __getattribute__')


user_next()
user_iter()
user_index()
user_getattr()
user_getattribute()
print('OK')
