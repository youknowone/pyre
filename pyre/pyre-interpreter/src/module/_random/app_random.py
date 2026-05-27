# app_random.py — `app_*` smoke test for the `py_module! appleveldefs:` arm.
# Mirrors `pypy/module/_random/app_random.py`'s role: pure-Python helpers that
# share `_random`'s namespace without per-helper Rust stub closures.


def _ascii_seed(s):
    """Sum the ASCII codes of `s`, used by tests as a deterministic int seed."""
    return sum(ord(c) for c in s)
