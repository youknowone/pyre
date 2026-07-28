# pyre-check: max-pypy-ratio=6
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
