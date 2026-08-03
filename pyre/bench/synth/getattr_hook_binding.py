# pyre-check: max-pypy-ratio=25
# objspace.py:710 get_and_call_function: a __getattr__ (or __getattribute__)
# defined as a classmethod or staticmethod must be bound through __get__ before
# being called, exactly like any other special method, so it receives the
# arguments the descriptor protocol gives it.


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


N = 30000


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
