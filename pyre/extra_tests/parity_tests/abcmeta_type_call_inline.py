# CPython-suite gap: `test_abc` never instantiates an ABCMeta-built class in a
# loop hot enough to compile, and no suite test installs `__call__` on a
# metaclass after such a loop has run.
# `typeobject.py:_type_call` runs unless the metatype supplies a `__call__` of
# its own.  An `ABCMeta` subclass supplies `__instancecheck__`,
# `__subclasscheck__` and `register` and leaves `__call__` alone, so a class
# built with one instantiates through the same path a plain class does.  The
# inline emit used to ask whether the metatype WAS `type`, which refused every
# such class -- and `Fraction`, `Decimal` and every `collections.abc` subclass
# with it.
#
# parity-tests reason: the admission is only sound while the answer holds, and
# what makes it hold is a pin on the METACLASS's version tag -- the class's own
# tag does not move when its metaclass gains an attribute.  A `__call__`
# installed on the metaclass mid-loop must take over on the next iteration, so
# this belongs where a stale answer is visible as a wrong number rather than as
# a missed optimisation.
#
# Each rebind happens INSIDE its loop: a call after the loop is interpreted and
# would not consult what the trace baked.
import abc

N = 40000
SWITCH = N // 2


class Meta(abc.ABCMeta):
    pass


class Point(metaclass=Meta):
    def __init__(self, x):
        self.x = x


class Fixed:
    x = 7


def inlines():
    # The plain shape: default `__new__`, an `__init__` the walk can enter, and
    # a metaclass that overrides neither.
    total = 0
    i = 0
    while i < N:
        total += Point(i).x
        i += 1
    assert total == N * (N - 1) // 2, 'wrong sum: %r' % (total,)


def metaclass_gains_call():
    total = 0
    i = 0
    while i < N:
        total += Point(i).x
        if i == SWITCH:
            Meta.__call__ = lambda cls, x: Fixed()
        i += 1
    # `i == SWITCH` is assigned before the rebind, so iterations 0..SWITCH read
    # their own index and the rest read `Fixed.x`.
    expected = SWITCH * (SWITCH + 1) // 2 + (N - SWITCH - 1) * 7
    assert total == expected, 'baked a stale metaclass __call__: %r != %r' % (
        total,
        expected,
    )


inlines()
metaclass_gains_call()
print('OK')
