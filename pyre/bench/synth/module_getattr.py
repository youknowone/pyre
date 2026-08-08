# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# module.py Module.descr_getattribute (PEP 562): after a normal attribute miss
# the module-level __getattr__ in the module's own dict is consulted, called
# with just the attribute name.  A present attribute does not trigger it.

import sys

PRESENT = 7


def __getattr__(name):
    return 'lazy:' + name


def main():
    m = sys.modules[__name__]
    # a present module global does not trigger the hook
    print('present', m.PRESENT)
    # a missing module attribute triggers the module-level __getattr__
    print('missing', m.missing)
    print('missing2', m.another_one)


main()
