# pyre-check: max-pypy-ratio=25
# objspace.py:710 get_and_call_function: a __getattr__ (or __getattribute__)
# defined as a classmethod or staticmethod must be bound through __get__ before
# being called, exactly like any other special method, so it receives the
# arguments the descriptor protocol gives it.
#
# The hook inlines against the version-tag and map pins that make the miss
# constant, instead of costing one opaque residual per access holding the
# whole `object_getattr_miss` walk plus a fresh frame.  The ratio reads
# 7.4x/10.4x/8.7x (dynasm/cranelift/wasm); the bound is twice the slowest of
# those (10.4x), rounded up to the next multiple of five.


class ClassmethodGetattr:
    @classmethod
    def __getattr__(cls, name):
        return 'cm:%s:%s' % (cls.__name__, name)


class StaticmethodGetattr:
    @staticmethod
    def __getattr__(name):
        return 'sm:%s' % name


class PlainGetattr:
    def __getattr__(self, name):
        return 'plain:%s' % name


# The loop has to outrun check.py's EXEC_TIME_FLOOR_S, below which pypy's
# execution cannot be separated from its own startup and the printed ratio is an
# artifact of the clamp rather than a measurement.  600000 iterations put pypy
# at ~0.05s: ten times the posix floor and three times the coarser windows one.
N = 600000


def main():
    # classmethod hook receives the class as its first bound argument
    print('classmethod', ClassmethodGetattr().nope)
    # staticmethod hook receives only the name
    print('staticmethod', StaticmethodGetattr().nope)
    # plain function hook stays bound to the instance
    print('plain', PlainGetattr().nope)

    cm = ClassmethodGetattr()
    sm = StaticmethodGetattr()
    plain = PlainGetattr()
    total = 0
    i = 0
    while i < N:
        if cm.nope == 'cm:ClassmethodGetattr:nope':
            total += 1
        if sm.nope == 'sm:nope':
            total += 1
        if plain.nope == 'plain:nope':
            total += 1
        i += 1
    print('loop', total)


main()
