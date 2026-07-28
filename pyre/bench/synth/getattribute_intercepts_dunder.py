# pyre-check: max-pypy-ratio=5
# descroperation.py:87 — a user-defined __getattribute__ slot intercepts EVERY
# attribute access, including the name '__getattribute__' itself; there is no
# special-casing that bypasses the custom slot for dunder names.


class C:
    def __getattribute__(self, name):
        return 'got:' + name


def main():
    c = C()
    # the custom slot handles its own name
    print('dunder', c.__getattribute__)
    # and any other name
    print('other', c.foo)
    print('class', c.__class__)


main()
