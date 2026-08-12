import _ast
import platform
import types

from testutils import assert_raises

ns = types.SimpleNamespace(a=2, b="Rust")

assert ns.a == 2
assert ns.b == "Rust"
with assert_raises(AttributeError):
    _ = ns.c


def _function_type_kwdefaults():
    def source(a, /, b, *, c):
        return a + b + c

    function = types.FunctionType(
        source.__code__, {}, "function", (1, 2), None, {"c": 3}
    )
    assert function() == 6
    assert function.__defaults__ == (1, 2)
    assert function.__kwdefaults__ == {"c": 3}

    with assert_raises(TypeError) as raised:
        types.FunctionType(source.__code__, {}, "function", None, None, 3)
    assert str(raised.exception) == "arg 6 (kwdefaults) must be None or dict"


_function_type_kwdefaults()


def _run_missing_type_params_regression():
    args = _ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    pass_stmt = _ast.Pass(lineno=1, col_offset=4, end_lineno=1, end_col_offset=8)
    fn = _ast.FunctionDef("f", args, [pass_stmt], [], None, None)
    fn.lineno = 1
    fn.col_offset = 0
    fn.end_lineno = 1
    fn.end_col_offset = 8
    mod = _ast.Module([fn], [])
    compiled = compile(mod, "<stdlib_types_missing_type_params>", "exec")
    exec(compiled, {})


_run_missing_type_params_regression()
