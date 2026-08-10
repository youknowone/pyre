"""CPython 3.14 text signatures for functions in the builtins module."""

import builtins
import inspect


EXPECTED = {
    "__import__": "($module, /, name, globals=None, locals=None, fromlist=(),\n           level=0)",
    "abs": "($module, x, /)",
    "aiter": "($module, async_iterable, /)",
    "all": "($module, iterable, /)",
    "anext": "($module, aiterator, default=<unrepresentable>, /)",
    "any": "($module, iterable, /)",
    "ascii": "($module, obj, /)",
    "bin": "($module, number, /)",
    "breakpoint": "($module, /, *args, **kws)",
    "callable": "($module, obj, /)",
    "chr": "($module, i, /)",
    "compile": "($module, /, source, filename, mode, flags=0,\n        dont_inherit=False, optimize=-1, *, _feature_version=-1)",
    "delattr": "($module, obj, name, /)",
    "divmod": "($module, x, y, /)",
    "eval": "($module, source, /, globals=None, locals=None)",
    "exec": "($module, source, /, globals=None, locals=None, *, closure=None)",
    "format": "($module, value, format_spec='', /)",
    "globals": "($module, /)",
    "hasattr": "($module, obj, name, /)",
    "hash": "($module, obj, /)",
    "hex": "($module, number, /)",
    "id": "($module, obj, /)",
    "input": "($module, prompt='', /)",
    "isinstance": "($module, obj, class_or_tuple, /)",
    "issubclass": "($module, cls, class_or_tuple, /)",
    "len": "($module, obj, /)",
    "locals": "($module, /)",
    "oct": "($module, number, /)",
    "open": "($module, /, file, mode='r', buffering=-1, encoding=None,\n     errors=None, newline=None, closefd=True, opener=None)",
    "ord": "($module, character, /)",
    "pow": "($module, /, base, exp, mod=None)",
    "print": "($module, /, *args, sep=' ', end='\\n', file=None, flush=False)",
    "repr": "($module, obj, /)",
    "round": "($module, /, number, ndigits=None)",
    "setattr": "($module, obj, name, value, /)",
    "sorted": "($module, iterable, /, *, key=None, reverse=False)",
    "sum": "($module, iterable, /, start=0)",
}

for name, signature in EXPECTED.items():
    assert getattr(builtins, name).__text_signature__ == signature, name

for name in ("__build_class__", "dir", "getattr", "iter", "max", "min", "next", "vars"):
    assert getattr(builtins, name).__text_signature__ is None, name

assert str(inspect.signature(len)) == "(obj, /)"
assert str(inspect.signature(sorted)) == (
    "(iterable, /, *, key=None, reverse=False)"
)
assert str(inspect.signature(open)) == (
    "(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, "
    "closefd=True, opener=None)"
)
assert str(inspect.signature(print)) == (
    "(*args, sep=' ', end='\\n', file=None, flush=False)"
)

print("OK")
