# A hot loop FOLLOWED by a `finally` (or nested `finally`) containing a
# bare `raise` (RAISE_VARARGS argc==0).  When such a bare re-raise is
# reached by normal fall-through — no `PUSH_EXC_INFO` seeded the handler
# exception — the RAISE_VARARGS(0) PC is uncovered and the FrameState
# carries no `last_exception` pair.  The codewriter previously routed this
# unconditionally to `emit_reraise!`, whose `exception_edge_extravars`
# asserts a materialized pair, so `transform_graph_to_jitcode` panicked
# ("exception edge state missing last_exception pair") while building the
# jitcode for any hot function shaped like this.  The parity-correct coding
# is the explicit `raise/r` of the current exception
# (`get_current_exception()` + raise), as the `Reraise` no-catch arm
# already does.  This bench pins that such functions compile and run
# byte-identically to CPython.
N = 2000000


def finally_bare_raise_cold(n):
    # Bare `raise` in a `finally` reached by fall-through, guarded off for
    # the tested input.  The value path (`return s`) executes, but the
    # codewriter must still lower the cold RAISE_VARARGS(0) edge.
    i = 0
    s = 0
    while i < n:
        s = s + i
        i = i + 1
    try:
        return s
    finally:
        if s < 0:
            raise
    return 0


def finally_bare_raise_fires(n):
    # Bare `raise` in a `finally` with no active exception actually fires:
    # RuntimeError ("No active exception to re-raise"), caught by the outer
    # handler.  Exercises the runtime `get_current_exception()` == null →
    # raise → RuntimeError path.
    i = 0
    while i < n:
        i = i + 1
    try:
        try:
            i = i + 0
        finally:
            raise
    except RuntimeError:
        return i - 7


def finally_bare_raise_reraises(n):
    # The try body raises and the `finally`'s bare `raise` re-raises the
    # in-flight exception, caught by the outer handler.  Here an active
    # exception IS present when the bare raise runs.
    i = 0
    while i < n:
        i = i + 1
    try:
        try:
            raise ValueError(i)
        finally:
            raise
    except ValueError as e:
        return e.args[0] + 3


def main():
    print(finally_bare_raise_cold(N))
    print(finally_bare_raise_fires(N))
    print(finally_bare_raise_reraises(N))


main()
