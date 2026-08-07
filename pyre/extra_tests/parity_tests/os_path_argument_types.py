"""A path argument is a str, a bytes, or an os.PathLike — nothing else.

A buffer is not a path. `bytearray` was accepted for one release cycle with a
DeprecationWarning and is now refused like any other type, so every one of these
boundaries reports a TypeError rather than addressing a file the caller never
named.

The message has two shapes. An entry point that converts its own argument names
itself, names the argument it turned away, and lists what that argument takes —
the list widens with `integer` exactly where the call can work on a descriptor,
so `stat` and `lstat` word it differently. The ones that convert a path on
someone else's behalf (`os.fspath`, `os.fsencode`) name no caller. Both report
the type by its own name, without the module that qualifies it in other
messages.

`os.startfile`, `os.listmounts`, Windows' `os.system` and the `_get*name`
family are left out: they exist on Windows alone, so their wording cannot be
measured against the oracle from here.
"""

import array
import atexit
import os
import shutil
import sys
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
atexit.register(shutil.rmtree, d, ignore_errors=True)
b = d.encode()
# A name that exists, for the boundaries that take a second path: the argument
# under test is the one that has to be refused, so the other has to be good.
# It is never opened — the conversion fails before any of these calls reaches
# the filesystem.
GOOD = os.path.join(d, "good")
open(GOOD, "wb").close()

# `array.array` is here for its name alone: the qualified `array.array` is what
# other messages report ("sequence item 0: expected str instance, array.array
# found"), and a path boundary reports the bare `array`.
BUFFERS = [
    ("bytearray", bytearray(b)),
    ("memoryview", memoryview(b)),
    ("array", array.array("B", b)),
]

# Every path-taking entry point, with whatever it wants after the path, and the
# message it words the rejection with. `path_converter` fills the function name
# and the argument name from the argument clinic, so the entry point names
# itself and says which of its arguments it turned away — `link` calls its two
# `src` and `dst` rather than both `path`.
PATH_ONLY = "string, bytes or os.PathLike"
WITH_FD = "string, bytes, os.PathLike or integer"
PINNED = {
    "stat": ((), f"stat: path should be {WITH_FD}, not {{}}"),
    # listdir names `integer` only where it can open a directory descriptor;
    # the Windows build has no fdopendir and leaves that word out.
    "listdir": (
        (),
        "listdir: path should be string, bytes, os.PathLike, integer or None, not {}"
        if sys.platform != "win32"
        else "listdir: path should be string, bytes, os.PathLike or None, not {}",
    ),
    "lchown": ((-1, -1), f"lchown: path should be {PATH_ONLY}, not {{}}"),
    "access": ((0,), f"access: path should be {PATH_ONLY}, not {{}}"),
    # `chdir` names `integer` for the same reason `listdir` does — it can
    # `fchdir`, and the Windows build cannot.
    "chdir": (
        (),
        f"chdir: path should be {WITH_FD}, not {{}}"
        if sys.platform != "win32"
        else f"chdir: path should be {PATH_ONLY}, not {{}}",
    ),
    "chmod": ((0o644,), f"chmod: path should be {WITH_FD}, not {{}}"),
    "chroot": ((), f"chroot: path should be {PATH_ONLY}, not {{}}"),
    "mkdir": ((), f"mkdir: path should be {PATH_ONLY}, not {{}}"),
    "open": ((0,), f"open: path should be {PATH_ONLY}, not {{}}"),
    "readlink": ((), f"readlink: path should be {PATH_ONLY}, not {{}}"),
    "remove": ((), f"remove: path should be {PATH_ONLY}, not {{}}"),
    "rmdir": ((), f"rmdir: path should be {PATH_ONLY}, not {{}}"),
    "scandir": (
        (),
        "scandir: path should be string, bytes, os.PathLike, integer or None, not {}"
        if sys.platform != "win32"
        else "scandir: path should be string, bytes, os.PathLike or None, not {}",
    ),
    "truncate": ((0,), f"truncate: path should be {WITH_FD}, not {{}}"),
    "unlink": ((), f"unlink: path should be {PATH_ONLY}, not {{}}"),
    "utime": ((), f"utime: path should be {WITH_FD}, not {{}}"),
    "mkfifo": ((), f"mkfifo: path should be {PATH_ONLY}, not {{}}"),
    "lstat": ((), f"lstat: path should be {PATH_ONLY}, not {{}}"),
    "statvfs": ((), f"statvfs: path should be {WITH_FD}, not {{}}"),
    "chown": ((-1, -1), f"chown: path should be {WITH_FD}, not {{}}"),
    "pathconf": (("PC_NAME_MAX",), f"pathconf: path should be {WITH_FD}, not {{}}"),
    "mknod": ((), f"mknod: path should be {PATH_ONLY}, not {{}}"),
    # The `l`-prefixed calls act on the link itself, so none of them can be
    # handed a descriptor and none names `integer`. chflags and its neighbours
    # are BSD's, absent elsewhere, and skipped by the getattr below.
    "lchmod": ((0o644,), f"lchmod: path should be {PATH_ONLY}, not {{}}"),
    "chflags": ((0,), f"chflags: path should be {PATH_ONLY}, not {{}}"),
    "lchflags": ((0,), f"lchflags: path should be {PATH_ONLY}, not {{}}"),
    # execv and execve name their path and nothing else: the argv entries and
    # the environment keys and values are converted on the sequence's or the
    # mapping's behalf, so those report the caller-less message.
    "execv": ((["a"],), f"execv: path should be {PATH_ONLY}, not {{}}"),
    "execve": ((["a"], {}), f"execve: path should be {PATH_ONLY}, not {{}}"),
}
# The boundaries that take two paths name them apart. `rename` and `replace`
# are one implementation here and two clinic declarations there, so each has to
# answer with its own name.
PAIRS = {
    "rename": ("src", "dst"),
    "replace": ("src", "dst"),
    "link": ("src", "dst"),
    "symlink": ("src", "dst"),
}
# posix_spawn and posix_spawnp share a body too, and the path they reject is
# named after whichever the caller reached.
SPAWN = ("posix_spawn", "posix_spawnp")

for name, buf in BUFFERS:
    for fn_name, (rest, message) in PINNED.items():
        fn = getattr(os, fn_name, None)
        if fn is None:
            continue
        rejects(lambda a: fn(a, *rest), buf, f"{fn_name}({name})", message.format(name))

    for fn_name, (first, second) in PAIRS.items():
        fn = getattr(os, fn_name, None)
        if fn is None:
            continue
        rejects(
            lambda a: fn(a, GOOD),
            buf,
            f"{fn_name}({name}, ...)",
            f"{fn_name}: {first} should be {PATH_ONLY}, not {name}",
        )
        rejects(
            lambda a: fn(GOOD, a),
            buf,
            f"{fn_name}(..., {name})",
            f"{fn_name}: {second} should be {PATH_ONLY}, not {name}",
        )

    for fn_name in SPAWN:
        fn = getattr(os, fn_name, None)
        if fn is None:
            continue
        rejects(
            lambda a: fn(a, ["x"], {}),
            buf,
            f"{fn_name}({name})",
            f"{fn_name}: path should be {PATH_ONLY}, not {name}",
        )

    # The ones that convert on someone else's behalf name no caller: the two
    # public converters, the POSIX `system`, and every element a sequence or a
    # mapping is walked for.
    for fn in (os.fsencode, os.fspath, os.fsdecode):
        rejects(
            fn,
            buf,
            f"{fn.__name__}({name})",
            f"expected str, bytes or os.PathLike object, not {name}",
        )

    UNNAMED = f"expected str, bytes or os.PathLike object, not {name}"
    if sys.platform != "win32":
        rejects(os.system, buf, f"system({name})", UNNAMED)
    rejects(
        lambda a: os.execv(GOOD, [a]),
        buf,
        f"execv(argv item {name})",
        UNNAMED,
    )
    # An environment *key* never reaches the converter: every buffer here is
    # mutable and so unhashable, and the dict turns it away first.
    rejects(
        lambda a: os.execve(GOOD, ["a"], {"k": a}),
        buf,
        f"execve(env value {name})",
        UNNAMED,
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
