# CPython-suite gap: `test_json` exercises `sort_keys=True` thoroughly and
# `test_importlib` blocks names with the `None` sentinel, but no test does both
# at once, so nothing notices an accelerator reaching a builtin by name.
#
# parity-tests reason: `sys.modules[name] = None` is the documented way a
# program stops everything downstream from importing that name, and the
# interpreter's own machinery is not downstream of it.  `encoder_listencode_dict`
# builds the items list and sorts it in place, so `json.dumps(d,
# sort_keys=True)` owes nothing to `sys.modules` -- an accelerator that reaches
# `builtins.sorted` instead fails on a module the program never asked it to use.
import json
import sys

saved = sys.modules["builtins"]
sys.modules["builtins"] = None
try:
    dumped = json.dumps({"b": 1, "a": 2, "c": 3}, sort_keys=True)
finally:
    sys.modules["builtins"] = saved

assert dumped == '{"a": 2, "b": 1, "c": 3}', dumped

# The keys are compared as the items tuples are, so an unorderable pair is a
# TypeError from the sort rather than from a lookup that never happened.
try:
    json.dumps({"a": 1, 2: 3}, sort_keys=True)
except TypeError:
    pass
else:
    raise AssertionError("mixed key types sorted")

print("OK")
