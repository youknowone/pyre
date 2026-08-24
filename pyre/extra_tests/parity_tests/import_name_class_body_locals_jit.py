# CPython-suite gap: the import-hook tests never put IMPORT_NAME inside a class
# body whose loop runs hot enough to compile, so nothing covers the one frame
# kind whose locals mapping is a real dict rather than None.
# parity-tests reason: `pyopcode.py:1119-1125` reads the frame's debug locals
# and substitutes None only when the frame has none.  A class body has one, so
# a traced IMPORT_NAME that bakes None is visible to any custom __import__.

import builtins

old_import = builtins.__import__
os_module = old_import("os")
seen = []


def hook(name, globals_arg, locals_arg, fromlist, level):
    seen.append(locals_arg is None)
    return os_module


N = 40000
builtins.__import__ = hook
try:

    class C:
        i = 0
        while i < N:
            import os

            i += 1

finally:
    builtins.__import__ = old_import

assert len(seen) == N, len(seen)
# A baked None shows up only once the loop compiles, so report the iteration
# it starts at rather than the whole list.
assert not any(seen), seen.index(True)
print("OK")
