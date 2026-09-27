# pyre-check: pypy-diverges: `type` statements are 3.12 syntax and t-strings
# are 3.14's; pypy3 reports a SyntaxError for both.
# CPython-suite gap: `test_global` reaches this only because its own import
# chain happens to run the first `type` statement with a nearly full nursery,
# and no test pairs a t-string with allocation pressure at all.
# parity-tests reason: `CALL_INTRINSIC_1`'s PEP 695 helpers live in `_typing`
# and `BUILD_TEMPLATE`'s in `_template`, and the first statement of each kind
# imports that module.  The operands are popped first, so while the module
# body runs they belong to no frame slot -- they have to travel through roots
# the collector forwards.
# parity-env: PYPY_GC_NURSERY=64k

"""Popped intrinsic operands outlive the import their helper needs.

The nursery is filled before each first-of-its-kind statement, so the import
it performs collects while its operands are live, and more churn afterwards
drives further collections over whatever the results kept.
"""


def churn(count):
    out = []
    for i in range(count):
        out.append((i, [i, i + 1], "s%d" % i))
    return out


before = churn(400)

type Alias = list[int]


class Generic[T]:
    def first(self, items: list[T]) -> T:
        return items[0]


def identity[T](value: T) -> T:
    return value


between = churn(400)

subject = ["payload", {"k": "v"}]
template = t"{subject!r:>4} and {len(before)}"

after = churn(4000)

assert Alias.__name__ == "Alias", Alias.__name__
assert Alias.__value__ == list[int], Alias.__value__
assert Generic.__type_params__[0].__name__ == "T"
assert Generic().first([7, 8]) == 7
assert identity.__type_params__[0].__name__ == "T"
assert identity(subject) is subject
assert list(template.strings) == ["", " and ", ""], list(template.strings)
assert [interpolation.value for interpolation in template.interpolations] == [
    subject,
    len(before),
]
assert [interpolation.conversion for interpolation in template.interpolations] == [
    "r",
    None,
]
assert [interpolation.format_spec for interpolation in template.interpolations] == [
    ">4",
    "",
]
assert len(before) == 400 and len(between) == 400 and len(after) == 4000
print("OK")
