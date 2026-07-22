import _opcode
import dis
import inspect


global_value = 42


def closure_target():
    return len([global_value])


instructions = list(dis.get_instructions(closure_target))
assert any(instruction.argval == "global_value" for instruction in instructions)
variables = inspect.getclosurevars(closure_target)
assert variables.globals == {"global_value": 42}
assert variables.builtins["len"] is len

for predicate in (
    _opcode.has_arg,
    _opcode.has_const,
    _opcode.has_name,
    _opcode.has_jump,
    _opcode.has_free,
    _opcode.has_local,
    _opcode.has_exc,
):
    try:
        predicate()
    except TypeError:
        pass
    else:
        raise AssertionError("opcode predicate accepted a missing opcode")

print("stdlib inspect ok")
