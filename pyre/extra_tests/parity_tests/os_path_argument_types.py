"""A path argument is a str, a bytes, or an os.PathLike — nothing else.

A buffer is not a path. `bytearray` was accepted for one release cycle with a
DeprecationWarning and is now refused like any other type, so every one of these
boundaries reports a TypeError rather than addressing a file the caller never
named.

The message has two shapes: an entry point that converts its own argument names
itself and lists what it takes, and the two that convert a path on someone
else's behalf (`os.fspath`, `os.fsencode`) name no caller. Both report the type
by its own name, without the module that qualifies it in other messages.
"""

import array
import os
import tempfile


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def rejects(fn, arg, what, message=None):
    try:
        fn(arg)
    except TypeError as e:
        if message is not None:
            check(str(e) == message, f"{what}: {e}")
    else:
        raise AssertionError(f"{what} did not raise")


d = tempfile.mkdtemp()
b = d.encode()

# `array.array` is here for its name alone: the qualified `array.array` is what
# other messages report ("sequence item 0: expected str instance, array.array
# found"), and a path boundary reports the bare `array`.
BUFFERS = [
    ("bytearray", bytearray(b)),
    ("memoryview", memoryview(b)),
    ("array", array.array("B", b)),
]

# Every path-taking entry point, with whatever it wants after the path. The
# message is pinned only for the boundaries that word it the same way on both
# interpreters today; the rest are asserted to reject, which is the property
# this file is about. (os.unlink and its neighbours name themselves on CPython
# and do not here — see the follow-up task on the unnamed path-only message.)
PINNED = {
    "stat": ((), "stat: path should be string, bytes, os.PathLike or integer, not {}"),
    "listdir": (
        (),
        "listdir: path should be string, bytes, os.PathLike, integer or None, not {}",
    ),
    "lchown": ((-1, -1), "lchown: path should be string, bytes or os.PathLike, not {}"),
}
UNPINNED = {
    "access": (0,),
    "chdir": (),
    "chmod": (0o644,),
    "mkdir": (),
    "open": (0,),
    "readlink": (),
    "remove": (),
    "rmdir": (),
    "scandir": (),
    "truncate": (0,),
    "unlink": (),
    "utime": (),
}

for name, buf in BUFFERS:
    for fn_name, (rest, message) in PINNED.items():
        fn = getattr(os, fn_name, None)
        if fn is None:
            continue
        rejects(lambda a: fn(a, *rest), buf, f"{fn_name}({name})", message.format(name))

    for fn_name, rest in UNPINNED.items():
        fn = getattr(os, fn_name, None)
        if fn is None:
            continue
        rejects(lambda a: fn(a, *rest), buf, f"{fn_name}({name})")

    # The two that convert on someone else's behalf name no caller.
    for fn in (os.fsencode, os.fspath):
        rejects(
            fn,
            buf,
            f"{fn.__name__}({name})",
            f"expected str, bytes or os.PathLike object, not {name}",
        )

# A buffer being refused must not have cost `bytes` its own arm.
check(os.stat(d).st_mode == os.stat(b).st_mode, "a bytes path stopped working")
check(os.fsencode(b) == b, "fsencode(bytes)")
check(os.fspath(b) == b, "fspath(bytes)")
check(sorted(os.listdir(b)) == sorted(n.encode() for n in os.listdir(d)), "listdir(bytes)")


class P:
    def __fspath__(self):
        return d


check(os.stat(P()).st_mode == os.stat(d).st_mode, "an os.PathLike stopped working")


# A buffer returned *by* __fspath__ is refused too, and there the message names
# the protocol rather than the argument.
class B:
    def __fspath__(self):
        return bytearray(b)


try:
    os.stat(B())
except TypeError as e:
    check(
        str(e) == "expected B.__fspath__() to return str or bytes, not bytearray",
        f"__fspath__ returning a buffer: {e}",
    )
else:
    raise AssertionError("__fspath__ returning a bytearray did not raise")

print("OK")
