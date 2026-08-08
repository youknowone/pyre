# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# module.py:143-162 — a module attribute miss with no module-level __getattr__
# reports the module's __name__ in the AttributeError message.

import sys


def main():
    m = sys.modules[__name__]
    try:
        m.does_not_exist
    except AttributeError as e:
        print('msg', str(e))


main()
