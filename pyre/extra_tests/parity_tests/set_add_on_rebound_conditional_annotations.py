# pyre-check: pypy-diverges: pyopcode.py SET_ADD is space.call_method(w_set, 'add', w_value), so a non-set raises AttributeError; this pins TypeError
# CPython-suite gap: test_type_annotations.RegressionTests.test_conditional_annotations_rebound
# expects TypeError when __conditional_annotations__ is rebound to a non-set.
# parity-tests reason: SET_ADD on that rebound receiver used to raise
# AttributeError ('int' object has no attribute 'add').

"""SET_ADD on a non-set raises TypeError."""

ns = {}
try:
    exec("__conditional_annotations__ = 0\nx: int", ns)
except TypeError:
    print("OK")
else:
    raise AssertionError("expected TypeError")
