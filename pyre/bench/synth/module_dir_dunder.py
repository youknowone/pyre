# pyre-check: max-pypy-ratio=5
# module.py:163 Module.descr_module__dir__ — a module-level __dir__ (stored in
# the module's own dict) drives dir(module); its result is sorted by dir().

import sys


def __dir__():
    return ['gamma', 'alpha', 'beta']


def main():
    print(dir(sys.modules[__name__]))


main()
