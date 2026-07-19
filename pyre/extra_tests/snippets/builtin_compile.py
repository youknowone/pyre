from testutils import assert_raises

# compile() basic mode acceptance
assert isinstance(
    compile("x = 1", "<test>", "exec"), type(compile("", "<test>", "exec"))
)
assert compile("1 + 1", "<test>", "eval") is not None
assert compile("1", "<test>", "single") is not None

# `optimize` accepts -1 (use config default), 0, 1, 2 only.
# Anything else raises ValueError with CPython's exact wording.
for ok in (-1, 0, 1, 2):
    compile("x = 1", "<test>", "exec", optimize=ok)


def _check_optimize_error(value):
    try:
        compile("x = 1", "<test>", "exec", optimize=value)
    except ValueError as e:
        assert str(e) == "compile(): invalid optimize value", repr(e)
    else:
        raise AssertionError(f"expected ValueError for optimize={value!r}")


for bad in (3, 4, 99, 255, 256, 1000, -2, -99, -128):
    _check_optimize_error(bad)

# Huge `optimize` values raise OverflowError during argument conversion,
# not ValueError. The exact wording differs from CPython here (Rust i32
# vs C int) — checking the type only, matching test_compile.py.
assert_raises(OverflowError, compile, "x = 1", "<test>", "exec", optimize=1 << 1000)


# Unrecognised `flags` bits raise ValueError. CPython uses British spelling
# ("unrecognised") so the message must match exactly.
def _check_flags_error(flags):
    try:
        compile("x = 1", "<test>", "exec", flags=flags)
    except ValueError as e:
        assert str(e) == "compile(): unrecognised flags", repr(e)
    else:
        raise AssertionError(f"expected ValueError for flags={flags!r}")


_check_flags_error(99999)
_check_flags_error(0x10000)


# PyPy's compile_to_ast path returns public, mutable `_ast` heap objects.
import ast

tree = compile("x = f'{value!r:>10}'", "<ast>", "exec", ast.PyCF_ONLY_AST)
assert isinstance(tree, ast.Module)
assign = tree.body[0]
assert assign.targets[0].id == "x"
formatted = assign.value.values[0]
assert formatted.value.id == "value"
assert formatted.conversion == ord("r")
assert formatted.format_spec.values[0].value == ">10"
assert compile(tree, "<ast>", "exec", ast.PyCF_ONLY_AST) is tree

tree = ast.parse(
    'match value:\n'
    '    case {"x": [first, *rest]} if rest:\n'
    '        pass\n'
)
case = tree.body[0].cases[0]
assert case.pattern.keys[0].value == "x"
assert case.pattern.patterns[0].patterns[0].name == "first"
assert case.pattern.patterns[0].patterns[1].name == "rest"
assert case.guard.id == "rest"
