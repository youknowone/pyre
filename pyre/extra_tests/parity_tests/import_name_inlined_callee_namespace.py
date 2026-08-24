# CPython-suite gap: `test_import` installs a replacement `__import__` and
# checks the arguments it receives, but only from module level -- never from a
# function hot enough to be traced, and never from one whose `__globals__` is a
# different mapping from its caller's.
# parity-tests reason: IMPORT_NAME reads the builtin `__import__`, `w_globals`
# and the locals through the tracer's frame variable, and a callee inlined into
# its caller's trace has to name its own frame there. Two comments in
# `codewriter.rs` disagree about whether it does: one says a non-portal callee's
# frame variable aliases the outermost frame, the other says every per-code
# jitcode carries its own. Only a callee whose globals are a distinct object can
# tell the two apart, and only while it is actually inlined.

"""An `import` inside an inlined callee names that callee's own namespace."""

import builtins

# Past `DEFAULT_THRESHOLD`, so most iterations run compiled rather than
# interpreted: an aliased frame would be baked into the recorded trace, and the
# check has to still be running when that trace executes.
WARM = 4000

# A callee whose `__globals__` is a mapping the caller's frame cannot reach, so
# `globals is CALLEE_GLOBALS` is a frame-identity test rather than a
# module-identity one.
CALLEE_GLOBALS = {"__name__": "inlined_importer", "__builtins__": builtins}
exec(compile(
    "def callee(tag):\n"
    "    own = tag & 7\n"
    "    import sys\n"
    "    return own + (sys is not None)\n",
    "<inlined_importer>", "exec"), CALLEE_GLOBALS)
callee = CALLEE_GLOBALS["callee"]

MAIN_GLOBALS = globals()
real_import = builtins.__import__
calls = 0
wrong_globals = []
wrong_locals = []


def spy(name, globals=None, locals=None, fromlist=(), level=0):
    global calls
    calls += 1
    if globals is not CALLEE_GLOBALS:
        wrong_globals.append(
            "caller" if globals is MAIN_GLOBALS else repr(globals)[:60])
    # An optimized frame binds no locals mapping, so both references hand over
    # None here -- after `locals()` and after `f_locals` alike. A frame
    # answering with some other scope's mapping shows up as one that is not
    # None rather than as a wrong namespace.
    if locals is not None:
        wrong_locals.append(repr(locals)[:60])
    return real_import(name, globals, locals, fromlist, level)


def outer(n):
    total = 0
    for i in range(n):
        total += callee(i)
    return total


builtins.__import__ = spy
try:
    result = outer(WARM)
finally:
    builtins.__import__ = real_import

# `__import__` runs on every iteration, module already in `sys.modules` or not,
# so a trace that hoisted or folded the lookup away shows up as a short count
# rather than as a wrong namespace.
assert calls == WARM, calls
assert not wrong_globals, wrong_globals[:4]
assert not wrong_locals, wrong_locals[:4]
assert result == sum((i & 7) + 1 for i in range(WARM)), result

print("OK")
