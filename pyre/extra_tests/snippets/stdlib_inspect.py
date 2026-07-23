import _opcode
import dis
import inspect
import opcode
import typing


global_value = 42


def closure_target():
    return len([global_value])


instructions = list(dis.get_instructions(closure_target))
assert any(instruction.argval == "global_value" for instruction in instructions)
variables = inspect.getclosurevars(closure_target)
assert variables.globals == {"global_value": 42}
assert variables.builtins["len"] is len

parameter = inspect.Parameter("value", inspect.Parameter.POSITIONAL_ONLY)
same_parameter = inspect.Parameter("value", inspect.Parameter.POSITIONAL_ONLY)
assert parameter == same_parameter
assert not parameter != same_parameter

signature = inspect.Signature([parameter])
bound = signature.bind(1)
same_bound = signature.bind(1)
assert bound == same_bound
assert not bound != same_bound


class EqualOnly:
    def __eq__(self, other):
        return isinstance(other, EqualOnly)


assert EqualOnly() == EqualOnly()
assert not EqualOnly() != EqualOnly()

comparison_calls = []


class ComparisonBase:
    def __eq__(self, other):
        comparison_calls.append("base")
        return NotImplemented


class ComparisonSubclass(ComparisonBase):
    def __eq__(self, other):
        comparison_calls.append("subclass")
        return True


assert ComparisonBase() == ComparisonSubclass()
assert comparison_calls == ["subclass"]

assert inspect.formatannotation(typing.List[str] | int) == "List[str] | int"
assert typing.Union[int] is int
assert repr(typing.Union[typing.List[str], int]) == "typing.List[str] | int"
try:
    object() | int
except TypeError:
    pass
else:
    raise AssertionError("an arbitrary object participated in union syntax")

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

assert _opcode.stack_effect(dis.opmap["NOP"], None) == 0
try:
    _opcode.stack_effect(dis.opmap["NOP"], 0, True)
except TypeError:
    pass
else:
    raise AssertionError("_opcode.stack_effect accepted positional jump")
assert opcode._intrinsic_1_descs[0] == "INTRINSIC_1_INVALID"
assert opcode._intrinsic_2_descs[0] == "INTRINSIC_2_INVALID"

print("stdlib inspect ok")
