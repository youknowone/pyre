# CPython-suite gap: test_re does not call _sre.template on a list subclass,
# a wide group index, or inspect the template object's type.
# parity-tests reason: _sre.template accepts list subclasses and ssize_t
# indexes, and returns an _sre.SRE_Template rather than a tuple.
# pyre-check: pypy-diverges: PyPy 3.11 re has no _compile_template and no _sre.SRE_Template; pypy3 raises AttributeError on re._compile_template.

"""_sre.template builds an SRE_Template and accepts wide indexes."""

import re
import _sre

pattern = re.compile("(a)")
template = re._compile_template(pattern, r"\1")
assert type(template).__name__ == "SRE_Template", type(template)
assert type(template).__module__ == "_sre", type(template).__module__
assert re.sub("(a)", r"\1\1", "aa") == "aaaa"


class MyList(list):
    pass


built = _sre.template(pattern, MyList(["", 1, ""]))
assert type(built).__name__ == "SRE_Template", type(built)


class I(int):
    pass


wide = _sre.template(pattern, ["", 2**40, ""])
assert type(wide).__name__ == "SRE_Template"
subclass_index = _sre.template(pattern, ["", I(1), ""])
assert type(subclass_index).__name__ == "SRE_Template"
bool_index = _sre.template(pattern, ["", True, ""])
assert type(bool_index).__name__ == "SRE_Template"

try:
    _sre.template(pattern, ["", 2**70, ""])
except OverflowError as error:
    assert str(error) == "Python int too large to convert to C ssize_t", error
else:
    raise AssertionError("2**70 must raise OverflowError")

try:
    _sre.template(pattern, ["", -(2**70), ""])
except OverflowError as error:
    assert str(error) == "Python int too large to convert to C ssize_t", error
else:
    raise AssertionError("-(2**70) must raise OverflowError")

try:
    _sre.template(pattern, ["", -1, ""])
except TypeError as error:
    assert str(error) == "invalid template", error
else:
    raise AssertionError("a negative index must be an invalid template")

try:
    _sre.template(pattern, ["", "x", ""])
except TypeError as error:
    assert str(error) == "an integer is required", error
else:
    raise AssertionError("a non-int index must require an integer")

try:
    _sre.template(pattern, ("", 1, ""))
except TypeError as error:
    assert "must be list" in str(error), error
else:
    raise AssertionError("a tuple must be refused")

print("OK")
