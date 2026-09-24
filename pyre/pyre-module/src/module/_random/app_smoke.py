# app_smoke.py — test-only `appleveldefs:` arm fixture for `macro_smoke`.
# Pure-Python helper sharing the smoke module namespace, mirroring the
# role of pypy/module/_random/app_random.py.


def _ascii_seed(s):
    """Sum the ASCII codes of s, used by tests as a deterministic int seed."""
    return sum(ord(c) for c in s)
