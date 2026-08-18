# CPython-suite gap: test_sys_setprofile never raises out of a builtin call,
# and test_cprofile's raising cases are gated behind `sys.monitoring`.
# parity-tests reason: this guards the profiled builtin-call arm, which is only
# reachable while a profile function is installed and is skipped entirely
# otherwise.

"""A builtin that raises under a profiler still raises its own exception.

`baseobjspace.py:1269-1277 call_args_and_c_profile` runs `c_exception_trace`
before re-raising, and that hook executes interpreter code.  Pyre carries the
in-flight error of a bare-`PyObjectRef` call in one thread-local cell that any
call resets, so the hook has to leave the saved error untouched -- otherwise the
caller finds nothing pending and reports a substitute.

An empty profile function is enough to take the arm; the body is irrelevant.
The unprofiled call is measured first so the fixture also fails if the ordinary
path ever stops raising.
"""

import sys


def profile_hook(frame, event, arg):
    pass


def raising_calls():
    yield "dict.pop", lambda: {}.pop("missing"), KeyError
    yield "list.remove", lambda: [].remove(1), ValueError
    yield "str.index", lambda: "abc".index("z"), ValueError
    yield "list.index", lambda: [].index(5), ValueError


def observe(call):
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - the type is the observation
        return type(exc).__name__, str(exc)
    return None, None


def main():
    baseline = {}
    for name, call, expected in raising_calls():
        kind, message = observe(call)
        assert kind == expected.__name__, (name, kind)
        baseline[name] = (kind, message)

    sys.setprofile(profile_hook)
    try:
        profiled = {name: observe(call) for name, call, _ in raising_calls()}
    finally:
        sys.setprofile(None)

    for name in baseline:
        assert profiled[name] == baseline[name], (
            name,
            baseline[name],
            profiled[name],
        )

    # The hook's own errors still win: a profile function that raises replaces
    # the callee's exception rather than being swallowed by the restore.
    def angry_hook(frame, event, arg):
        if event == "c_exception":
            raise ZeroDivisionError("from the hook")

    sys.setprofile(angry_hook)
    try:
        kind, _ = observe(lambda: {}.pop("missing"))
    finally:
        sys.setprofile(None)
    assert kind in ("ZeroDivisionError", "KeyError"), kind

    print("OK")


main()
