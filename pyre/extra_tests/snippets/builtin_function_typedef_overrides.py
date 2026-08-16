# pyre-check: gate=1
# The `__doc__` slot routes through `getset_func_doc` which falls
# back to `BuiltinCode.getdocstring` (function.py:446-449). pyre's
# `len` is registered without a docstring so the access path
# returns whatever code.getdocstring yields — the test only checks
# that the lookup does not crash and that mutation/deletion fire
# the orthodox `_check_code_mutable` AttributeError per
# function.py:387 ("Cannot change __doc__ attribute of builtin
# functions").
doc_value = len.__doc__
self_is_module = type(len.__self__).__name__ == 'module'
repr_result = len.__repr__()
new_err = ''
try:
    type(len)()
except TypeError as e:
    new_err = str(e)
set_err = ''
try:
    len.__doc__ = 'x'
except AttributeError as e:
    set_err = str(e)
del_err = ''
try:
    del len.__doc__
except AttributeError as e:
    del_err = str(e)

assert self_is_module is True
assert repr_result == '<built-in function len>'
assert new_err == "cannot create 'builtin_function_or_method' instances"
assert '__doc__' in set_err
assert '__doc__' in del_err
