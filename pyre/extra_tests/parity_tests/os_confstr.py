"""os.confstr answers the host's configuration strings, and confstr_names names them.

Both were stubs answering None, so `confstr_names` was empty and every call
answered None whatever it was asked. The table is the host's own — the Apple
targets number `_CS_PATH` 1 and glibc numbers it 0 — so what is checked here is
what holds on any host that has the call: the names resolve, the values are
distinct, CS_PATH is a real search path, and an unknown name is refused.
"""

import os
import sys


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def raises(call, exc):
    try:
        call()
    except exc:
        return
    raise AssertionError(f"{exc.__name__} was not raised")


if sys.platform == "win32":
    for name in ("confstr", "confstr_names"):
        check(not hasattr(os, name), f"windows grew an os.{name}")
    print("OK")
    raise SystemExit

names = os.confstr_names
check(isinstance(names, dict), f"confstr_names is {type(names).__name__}")
check(bool(names), "confstr_names is empty")
check(all(isinstance(k, str) for k in names), "a confstr name is not a str")
check(all(isinstance(v, int) for v in names.values()), "a confstr value is not an int")
# The table maps each name onto its own number; two names sharing one would send
# two questions to the same place.
check(len(set(names.values())) == len(names), "two confstr names share a value")
check(all(k.startswith("CS_") for k in names), "a confstr name is not spelled CS_*")

# CS_PATH is the one entry every host with `confstr` defines: the search path
# for the standard utilities, which is a non-empty list of absolute directories.
check("CS_PATH" in names, "confstr_names has no CS_PATH")
path = os.confstr("CS_PATH")
check(isinstance(path, str), f"confstr('CS_PATH') answered {path!r}")
check(path, "confstr('CS_PATH') is empty")
for part in path.split(os.pathsep):
    check(part.startswith("/"), f"CS_PATH entry is not absolute: {part!r}")

# The number and the name are the same question.
check(os.confstr(names["CS_PATH"]) == path, "confstr(int) and confstr(str) disagree")

# Every name in the table is one the host answers: a str the host has a value
# for, or None where it defines the name but has no string for it. What none of
# them may be is an error.
for name in names:
    value = os.confstr(name)
    check(value is None or isinstance(value, str), f"confstr({name!r}) answered {value!r}")

# A name the table does not carry is refused rather than silently answered.
raises(lambda: os.confstr("CS_NOT_A_REAL_NAME"), ValueError)
raises(lambda: os.confstr(None), TypeError)
raises(lambda: os.confstr(2**40), OverflowError)

print("OK")
