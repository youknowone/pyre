# pyre-check: max-pypy-ratio=5
# pyre-check: min-pypy-ratio=0.45
# A lone-surrogate attribute name reaches the same __getattr__ fallbacks as a
# normal name: a module-level __getattr__ (PEP 562) and a classmethod hook that
# must be bound through __get__ before being called.

import sys


def __getattr__(name):
    return 'mod_hook:' + name


class WithClassmethodHook:
    @classmethod
    def __getattr__(cls, name):
        return cls.__name__ + ':' + name


def main():
    surr = chr(0xDC80)
    # module-level __getattr__ fires for a surrogate name
    r = getattr(sys.modules[__name__], surr)
    print('mod_surrogate', r == 'mod_hook:' + surr)
    # a classmethod __getattr__ binds the class, even for a surrogate name
    r2 = getattr(WithClassmethodHook(), surr)
    print('cm_surrogate', r2 == 'WithClassmethodHook:' + surr)


main()
