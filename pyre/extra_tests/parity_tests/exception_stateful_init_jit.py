# CPython-suite gap: exception tests do not inline stateful construction in a trace.
# parity-tests reason: this guards pyre's JIT exception reconstruction.

"""A hot exception constructor keeps the fields its `descr_init` writes.

The traced inline constructor rebuilds an exception from `kind` / `w_class` /
`args_w` alone, so a kind whose `descr_init` stores extra flattened fields must
fall back to the runtime constructor.  Each class below is built often enough
for that path to be reached, and the check reads a field `args_w` cannot carry.
"""

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=1,function_threshold=1")

ROUNDS = 4000


def make_syntax():
    return SyntaxError("m", ("f.py", 1, 2, "src"))


def make_stop(value):
    return StopIteration(value)


def make_attribute():
    return AttributeError("a", name="nm", obj=None)


def make_name():
    return NameError("n", name="who")


def make_system_exit():
    return SystemExit(3)


def make_import():
    return ImportError("i", name="mod", path="/p")


def make_group():
    return ExceptionGroup("g", [ValueError("v")])


for round_index in range(ROUNDS):
    error = make_syntax()
    assert error.msg == "m", (round_index, error.msg)
    assert error.filename == "f.py", (round_index, error.filename)
    assert error.lineno == 1, (round_index, error.lineno)
    assert error.offset == 2, (round_index, error.offset)
    assert error.text == "src", (round_index, error.text)

    # `value` is independent of `args`, so reassigning `args` exposes a
    # reconstruction that only carried the argument list.
    stop = make_stop(round_index)
    stop.args = (-1,)
    assert stop.value == round_index, (round_index, stop.value)

    attribute = make_attribute()
    assert attribute.name == "nm", (round_index, attribute.name)

    name_error = make_name()
    assert name_error.name == "who", (round_index, name_error.name)

    system_exit = make_system_exit()
    system_exit.args = ()
    assert system_exit.code == 3, (round_index, system_exit.code)

    import_error = make_import()
    assert import_error.name == "mod", (round_index, import_error.name)
    assert import_error.path == "/p", (round_index, import_error.path)

    group = make_group()
    assert group.message == "g", (round_index, group.message)
    assert len(group.exceptions) == 1, (round_index, group.exceptions)

print("OK")
