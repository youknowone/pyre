"""Import keeps a lone-surrogate module name wrapped.

PyPy's ``interp___import__`` passes ``w_name`` to importlib rather than
forcing it through a host UTF-8 string.  A failed lookup must therefore be an
ordinary ModuleNotFoundError, not an interpreter panic.
"""

name = "nonencodable\udbff"

try:
    __import__(name)
except ModuleNotFoundError as exc:
    assert exc.name == name
    assert exc.msg == "No module named 'nonencodable\\udbff'"
    assert str(exc) == exc.msg
    assert exc.args == (exc.msg,)
else:
    raise AssertionError("surrogate-bearing module name unexpectedly imported")

print("OK")
