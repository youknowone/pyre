"""RERAISE must pop the exception above the saved ``lasti`` value.

PyPy's frame stack is authoritative at ``RERAISE``: ``popvalue()`` reads the
live top slot, while ``peekvalue(oparg)`` only reads the saved instruction
offset below it.  The JIT must not substitute the symbolic ``lasti`` box for
that top-of-stack exception when the cleanup shape is
``COPY 3; POP_EXCEPT; RERAISE 1``.
"""

import inspect

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=3,function_threshold=3")

for _ in range(20):
    try:
        inspect.Parameter("bar", kind=5, default=42)
    except ValueError as exc:
        assert "value 5 is not a valid Parameter.kind" in str(exc)
    else:
        raise AssertionError("invalid Parameter.kind did not raise ValueError")

print("OK")
