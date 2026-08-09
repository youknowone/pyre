# pyre-check: max-pypy-ratio=20
# A class body and an `exec(code, ns)` body are the two frames that receive
# their namespace through `setdictscope`, and both route a hot loop through the
# JIT portal.
#
# What this file gates in a release build is the namespace read: `exec` accepts
# any namespace that passes `PyDict_Check`, so the dict subclass and the
# OrderedDict below are both legal, and neither carries the raw `W_DictObject`
# layout that the tracer read namespace lengths and callee globals through.
# Reading them that way faulted, and this workload segfaults without the
# guards.  A fresh namespace per repeat also has the recorded trace looked up
# again against a length that cannot be read.
#
# These frames also read a Ref static virtualizable field for LOAD_NAME /
# STORE_NAME, which is what makes them the shapes that reach
# `check_synchronized_virtualizable`; a function-frame loop reads no Ref vable
# field and never looks.  Mirroring a static vable field into the shadow
# without flushing it back left the live frame behind and tripped that check on
# `last_instr` -- but the check is `debug_assertions`-gated and the values stay
# correct without it, so that half is not observable here.  It needs a debug
# build to show, and nothing in this suite runs one.
#
# The loop has to run far enough in one entry to warm the portal.
# `exec_fresh_globals_delete_name` already runs an exec-body loop and stayed
# green throughout: at 400 iterations it never reached the divergence.
import collections

N = 3000
REPEAT = 5


def class_body_while():
    class C:
        i = 0
        total = 0
        while i < N:
            total += i
            i += 1

    return C.total


def class_body_for():
    class C:
        total = 0
        for i in range(N):
            total += i

    return C.total


CODE = compile(
    "total = 0\ni = 0\nwhile i < N:\n    total += i\n    i += 1\n",
    "<exec-ns>",
    "exec",
)


class NS(dict):
    pass


def exec_into(ns):
    ns["N"] = N
    exec(CODE, ns)
    return ns["total"]


def main():
    checksum = 0
    for _ in range(REPEAT):
        checksum += class_body_while()
        checksum += class_body_for()
        checksum += exec_into(NS())
        checksum += exec_into(collections.OrderedDict())
    print(checksum)


main()
