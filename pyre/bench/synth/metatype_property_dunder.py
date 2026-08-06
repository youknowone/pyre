# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# typeobject.py:811-819 — a metaclass data descriptor (property) named like a
# hardcoded type attribute wins over the built-in one; an un-overridden dunder
# still resolves through the built-in path.


class Meta(type):
    @property
    def __name__(cls):
        return 'custom:' + cls._real


class C(metaclass=Meta):
    _real = 'C'


def main():
    # the metaclass property beats the built-in __name__
    print('overridden', C.__name__)
    # __mro__ has no metaclass override -> built-in resolution
    print('mro_len', len(C.__mro__))
    # an ordinary class attribute still resolves
    print('attr', C._real)


main()
