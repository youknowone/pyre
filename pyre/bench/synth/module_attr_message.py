# pyre-check: max-pypy-ratio=14
# The ceiling is twice the slowest ratio observed, 6.6x on the macos runner;
# the gate it replaces sat inside the run-to-run spread.
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
