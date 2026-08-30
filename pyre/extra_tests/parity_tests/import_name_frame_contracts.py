# CPython-suite gap: import-hook tests do not run IMPORT_NAME hot across a
# hook rebind, a class namespace, or an inlined callee with foreign globals.
# parity-tests reason: traced IMPORT_NAME must retain its co_names object and
# read the live hook, globals, and locals from the frame that owns the opcode.

import builtins


REAL_IMPORT = builtins.__import__
OS_MODULE = REAL_IMPORT("os")


def cached_name_and_live_hook():
    calls = [0, 0]
    total = 40000
    switch = total // 2

    def run():
        names = run.__code__.co_names
        expected_name = names[names.index("os")]
        expected_globals = globals()

        def first(name, globals_arg, locals_arg, fromlist, level):
            assert name is expected_name
            assert globals_arg is expected_globals and locals_arg is None
            assert fromlist is None and level == 0
            calls[0] += 1
            return OS_MODULE

        def second(name, globals_arg, locals_arg, fromlist, level):
            assert name is expected_name
            assert globals_arg is expected_globals and locals_arg is None
            assert fromlist is None and level == 0
            calls[1] += 1
            return OS_MODULE

        builtins.__import__ = first
        try:
            for i in range(total):
                import os

                assert os is OS_MODULE
                if i == switch:
                    builtins.__import__ = second
        finally:
            builtins.__import__ = REAL_IMPORT

    run()
    assert calls == [switch + 1, total - switch - 1], calls


def class_body_keeps_its_locals():
    seen = []

    def hook(name, globals_arg, locals_arg, fromlist, level):
        seen.append(locals_arg is None)
        return OS_MODULE

    builtins.__import__ = hook
    try:
        class Imported:
            i = 0
            while i < 40000:
                import os

                i += 1
    finally:
        builtins.__import__ = REAL_IMPORT

    assert Imported.i == 40000
    assert len(seen) == 40000 and not any(seen), seen[:4]


CALLEE_GLOBALS = {"__name__": "inlined_importer", "__builtins__": builtins}
exec(
    compile(
        "def callee(tag):\n"
        "    own = tag & 7\n"
        "    import sys\n"
        "    return own + (sys is not None)\n",
        "<inlined_importer>",
        "exec",
    ),
    CALLEE_GLOBALS,
)
callee = CALLEE_GLOBALS["callee"]
MAIN_GLOBALS = globals()
inlined_calls = 0
wrong_globals = []
wrong_locals = []


def spy(name, globals=None, locals=None, fromlist=(), level=0):
    global inlined_calls
    inlined_calls += 1
    if globals is not CALLEE_GLOBALS:
        wrong_globals.append("caller" if globals is MAIN_GLOBALS else repr(globals)[:60])
    if locals is not None:
        wrong_locals.append(repr(locals)[:60])
    return REAL_IMPORT(name, globals, locals, fromlist, level)


def outer(n):
    total = 0
    for i in range(n):
        total += callee(i)
    return total


cached_name_and_live_hook()
class_body_keeps_its_locals()
builtins.__import__ = spy
try:
    result = outer(4000)
finally:
    builtins.__import__ = REAL_IMPORT

assert inlined_calls == 4000, inlined_calls
assert not wrong_globals, wrong_globals[:4]
assert not wrong_locals, wrong_locals[:4]
assert result == sum((i & 7) + 1 for i in range(4000)), result
print("OK")
