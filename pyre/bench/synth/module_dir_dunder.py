# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# module.py:163 Module.descr_module__dir__ — a module-level __dir__ (stored in
# the module's own dict) drives dir(module); its result is sorted by dir().

import sys


def __dir__():
    return ['gamma', 'alpha', 'beta']


def main():
    print(dir(sys.modules[__name__]))


main()
