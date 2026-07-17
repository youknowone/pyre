import types


EXPECTED = {
    "__doc__",
    "__eq__",
    "__ge__",
    "__gt__",
    "__hash__",
    "__le__",
    "__lt__",
    "__ne__",
    "__new__",
    "__replace__",
    "__repr__",
    "__sizeof__",
    "_co_code_adaptive",
    "_varname_from_oparg",
    "co_argcount",
    "co_branches",
    "co_cellvars",
    "co_code",
    "co_consts",
    "co_exceptiontable",
    "co_filename",
    "co_firstlineno",
    "co_flags",
    "co_freevars",
    "co_kwonlyargcount",
    "co_lines",
    "co_linetable",
    "co_lnotab",
    "co_name",
    "co_names",
    "co_nlocals",
    "co_positions",
    "co_posonlyargcount",
    "co_qualname",
    "co_stacksize",
    "co_varnames",
    "replace",
}

assert set(types.CodeType.__dict__) == EXPECTED
assert types.CodeType.__doc__ == "Create a code object.  Not for the faint of heart."


def sample(a, /, b=2, *, c=3):
    if a:
        return a + b + c
    return 0


code = sample.__code__
assert code.co_argcount == 2
assert code.co_posonlyargcount == 1
assert code.co_kwonlyargcount == 1
assert code.co_nlocals == len(code.co_varnames)
assert code.co_varnames[:3] == ("a", "b", "c")
assert code.co_name == "sample"
assert code.co_qualname == "sample"
assert isinstance(code.co_code, bytes)
assert isinstance(code._co_code_adaptive, bytes)
assert len(code._co_code_adaptive) == len(code.co_code)
assert isinstance(code.co_consts, tuple)
assert isinstance(code.co_names, tuple)
assert isinstance(code.co_freevars, tuple)
assert isinstance(code.co_cellvars, tuple)
assert isinstance(code.co_linetable, bytes)
assert isinstance(code.co_lnotab, bytes)
assert code.co_lnotab
assert isinstance(code.co_exceptiontable, bytes)
assert code._varname_from_oparg(0) == "a"

try:
    code._varname_from_oparg(-1)
except IndexError as exc:
    assert str(exc) == "tuple index out of range"
else:
    raise AssertionError("negative local index was accepted")

positions = list(code.co_positions())
lines = list(code.co_lines())
branches = list(code.co_branches())
assert len(positions) == len(code.co_code) // 2
assert lines and all(len(row) == 3 for row in lines)
assert branches == [(12, 18, 48)]


def loop(values):
    for value in values:
        pass


assert list(loop.__code__.co_branches()) == [(6, 10, 20)]

large_namespace = {}
large_body = "\n".join(f"        value = {number}" for number in range(300))
exec(
    "def large_branch(flag):\n"
    "    if flag:\n"
    f"{large_body}\n"
    "    return value\n",
    large_namespace,
)
large_branches = list(large_namespace["large_branch"].__code__.co_branches())
assert large_branches
assert max(max(row) for row in large_branches) > 512

same = code.replace()
same_dunder = code.__replace__()
assert same is not code
assert same == code
assert same_dunder == code
assert hash(same) == hash(code)
assert code.replace(co_name="renamed").co_name == "renamed"
assert code.replace(co_qualname="outer.renamed").co_qualname == "outer.renamed"
different_lines = code.replace(co_linetable=b"")
different_exceptions = code.replace(co_exceptiontable=b"x")
assert different_lines != code
assert different_exceptions != code
assert hash(different_lines) != hash(code)
assert hash(different_exceptions) != hash(code)
assert list(different_lines.co_positions()) != positions

for changes in (
    {"co_argcount": len(code.co_varnames) + 1},
    {"co_posonlyargcount": code.co_argcount + 1},
    {"co_kwonlyargcount": len(code.co_varnames) + 1},
    {"co_nlocals": code.co_nlocals + 1},
):
    try:
        code.replace(**changes)
    except (ValueError, SystemError):
        pass
    else:
        raise AssertionError(f"code.replace accepted inconsistent counts: {changes}")

for field in ("co_argcount", "co_stacksize", "co_flags", "co_firstlineno"):
    try:
        code.replace(**{field: 1 << 40})
    except OverflowError:
        pass
    else:
        raise AssertionError(f"code.replace accepted an oversized {field}")


def constants_sample():
    return 1j, frozenset({1, 2})


constants_code = constants_sample.__code__
assert constants_code.replace(co_consts=constants_code.co_consts) == constants_code


def outer(captured):
    def inner():
        return captured

    return inner


outer_code = outer.__code__
assert outer_code._varname_from_oparg(0) == "captured"
try:
    outer_code._varname_from_oparg(len(outer_code.co_varnames))
except IndexError:
    pass
else:
    raise AssertionError("cellvar/local alias was exposed as a second locals-plus slot")

assert types.CodeType.__eq__(code, object()) is NotImplemented
assert types.CodeType.__ne__(code, object()) is NotImplemented
for name in ("__lt__", "__le__", "__gt__", "__ge__"):
    assert getattr(types.CodeType, name)(code, same) is NotImplemented

assert repr(code).startswith("<code object sample at 0x")
assert 'file "' in repr(code)
assert code.__sizeof__() > 0

code_args = (
    code.co_argcount,
    code.co_posonlyargcount,
    code.co_kwonlyargcount,
    code.co_nlocals,
    code.co_stacksize,
    code.co_flags,
    code.co_code,
    code.co_consts,
    code.co_names,
    code.co_varnames,
    code.co_filename,
    code.co_name,
    code.co_qualname,
    code.co_firstlineno,
    code.co_linetable,
    code.co_exceptiontable,
    code.co_freevars,
    code.co_cellvars,
)
rebuilt = types.CodeType(*code_args)
assert rebuilt == code
assert rebuilt.co_lnotab == code.co_lnotab
assert list(rebuilt.co_lines()) == lines
assert list(rebuilt.co_positions()) == positions
rebuilt_function = types.FunctionType(rebuilt, globals())
assert rebuilt_function(4, 5, c=6) == 15

negative_line_args = list(code_args)
negative_line_args[13] = -1
negative_line = types.CodeType(*negative_line_args)
assert negative_line.co_firstlineno == -1

try:
    code.replace(unknown=1)
except TypeError as exc:
    assert str(exc) == "replace() got an unexpected keyword argument 'unknown'"
else:
    raise AssertionError("code.replace accepted an unknown field")

print("OK")
