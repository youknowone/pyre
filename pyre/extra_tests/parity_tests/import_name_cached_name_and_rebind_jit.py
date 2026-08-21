# CPython-suite gap: import-hook tests do not combine co_names identity with a
# hot loop that rebinds builtins.__import__ after IMPORT_NAME has been traced.
# parity-tests reason: IMPORT_NAME must keep PyPy's co_names_w object and live
# builtin lookup when its call is exposed to the meta-tracer.

import builtins


old_import = builtins.__import__
os_module = old_import("os")
calls = [0, 0]
N = 40000
SWITCH = N // 2


def run():
    names = run.__code__.co_names
    expected_name = names[names.index("os")]
    expected_globals = globals()

    def first(name, globals_arg, locals_arg, fromlist, level):
        assert name is expected_name
        assert globals_arg is expected_globals
        assert locals_arg is None
        calls[0] += 1
        return os_module

    def second(name, globals_arg, locals_arg, fromlist, level):
        assert name is expected_name
        assert globals_arg is expected_globals
        assert locals_arg is None
        calls[1] += 1
        return os_module

    builtins.__import__ = first
    try:
        i = 0
        while i < N:
            import os

            assert os is os_module
            if i == SWITCH:
                builtins.__import__ = second
            i += 1
    finally:
        builtins.__import__ = old_import


run()
assert calls == [SWITCH + 1, N - SWITCH - 1], calls
print("OK")
