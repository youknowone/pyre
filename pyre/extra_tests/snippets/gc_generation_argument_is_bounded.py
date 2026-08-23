# pyre-check: gate=1
"""`gc.collect` and `gc.get_objects` bound their generation argument.

The module reports three generations — `get_threshold` answers a three-tuple
and `set_threshold` writes three slots — so a generation outside `range(3)` is
an error, the way `gc_collect_impl` and `gc_get_objects_impl` reject one.
`get_objects` additionally takes `-1`, and `None`, for every generation at
once, and rejects anything below `-1` with its own message.
"""

import gc
import sys

# Every generation the module reports is accepted, and the answer is the count
# `gc.collect` documents rather than the generation echoed back.
for generation in (0, 1, 2):
    assert isinstance(gc.collect(generation), int), generation

for bad in (3, 4, 99, -1, -2):
    try:
        gc.collect(bad)
    except ValueError as exc:
        assert str(exc) == "invalid generation", (bad, str(exc))
    else:
        raise AssertionError(f"gc.collect({bad}) did not raise")

# `get_objects` takes the same three, plus the two spellings of "all of them".
for generation in (0, 1, 2, -1, None):
    assert isinstance(gc.get_objects(generation), list), generation
assert isinstance(gc.get_objects(), list)

try:
    gc.get_objects(3)
except ValueError as exc:
    assert "available generations (3)" in str(exc), str(exc)
else:
    raise AssertionError("gc.get_objects(3) did not raise")

try:
    gc.get_objects(-2)
except ValueError as exc:
    assert str(exc) == "generation parameter cannot be negative", str(exc)
else:
    raise AssertionError("gc.get_objects(-2) did not raise")

# The range check runs after the audit event, so a rejected generation is still
# reported to a hook.  Only the event is asserted, not what it carries: the
# audited generation is the argument in `gc_get_objects_impl` and the constant
# -1 in `referents.py:get_objects`, and this one keeps the constant.
audited = []
sys.addaudithook(lambda event, args: audited.append(event))

try:
    gc.get_objects(3)
except ValueError:
    pass
else:
    raise AssertionError("gc.get_objects(3) did not raise")
assert "gc.get_objects" in audited, audited
