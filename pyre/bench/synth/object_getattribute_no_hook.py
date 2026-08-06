# pyre-check: max-pypy-ratio=16
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest the macos
# runner and this machine observe (7.8x), rounded up. It read 5, a blanket
# value from the sweep that first gave every fixture a ceiling.
# descroperation.py:88 vs :234 — the bare object.__getattribute__ slot raises
# AttributeError on miss and does NOT consult __getattr__; normal attribute
# access (space.getattr) does.  Holds for instance and class/metaclass
# receivers alike.


class Meta(type):
    def __getattr__(cls, name):
        return 'meta_hook:' + name


class C(metaclass=Meta):
    pass


class WithHook:
    def __getattr__(self, name):
        return 'inst_hook:' + name


def attempt(fn):
    try:
        return fn()
    except AttributeError:
        return 'AttributeError'


def main():
    # normal class access consults the metaclass __getattr__
    print('class_normal', C.missing)
    # the bare slot does not
    print('class_bare', attempt(lambda: object.__getattribute__(C, 'missing')))
    # normal instance access consults __getattr__
    print('inst_normal', WithHook().missing)
    # the bare slot does not
    print('inst_bare', attempt(lambda: object.__getattribute__(WithHook(), 'missing')))


main()
