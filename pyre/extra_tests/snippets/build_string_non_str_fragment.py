# pyre-check: gate=1
import ast


def compile_joined(values):
    tree = ast.JoinedStr(values=values)
    return compile(ast.fix_missing_locations(ast.Expression(tree)), "<t>", "eval")


def assert_type_error(values, message):
    code = compile_joined(values)
    try:
        eval(code)
    except TypeError as exc:
        assert str(exc) == message, str(exc)
    else:
        raise AssertionError("expected TypeError: " + message)


assert_type_error(
    [ast.Constant("a"), ast.Constant(1)],
    "sequence item 1: expected str instance, int found",
)
assert_type_error(
    [ast.Constant(1), ast.Constant("a")],
    "sequence item 0: expected str instance, int found",
)


class S(str):
    pass


# Constant rejects a str subclass at compile time.  BUILD_STRING still
# accepts one: swap the middle const of an otherwise identical code object.
code = compile_joined([ast.Constant("a"), ast.Constant("b"), ast.Constant("c")])
code = code.replace(co_consts=("a", S("b"), "c"))
assert eval(code) == "abc", eval(code)

s = "z"
acc = []
for i in range(5000):
    acc.append(f"{i}-{s}")
assert acc[0] == "0-z"
assert acc[4999] == "4999-z"
assert len(acc) == 5000
