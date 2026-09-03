# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=warm
# Self-checking regression guard for the operands `exec` / `eval` thread from
# the code object's construction to the frame that runs it.
#
# `pyopcode.py:773-774` plants `__builtins__` with
# `space.call_method(w_globals, 'setdefault', ...)` and `compiling.py:109-110`
# with `space.contains_w` + `space.setitem_str`, all dispatched on the caller's
# own mapping so a dict subclass's override wins.  That override is user
# Python, so it can collect -- and at that point `exec_or_eval` is still
# holding the code object it compiled a few lines earlier, the namespace
# arguments, and a raw pointer interior to the code object's payload.
#
# Nothing else references a code object compiled from a string, so a collection
# inside that window reclaimed it: `createframe_obj` then stored globals
# through a dead header (`w_code_frame_stores_global` reaching the write
# barrier with a freed parent), or the frame ran against a stale namespace and
# the assignment the source performs was never visible to the caller.
#
# `frame_locals_snapshot`, `function_new_with_closure` and `createframe_obj`
# each allocate after that window too, which is why the operands are read back
# from their slots at every use rather than once after the plant.
#
# Every expectation below is the value CPython 3.14 and PyPy both produce.
import gc

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

failures = []


def churn():
    """Allocation the collection below has something to reclaim."""
    return [{} for _ in range(200)]


def collect():
    churn()
    gc.collect()


class SetdefaultNS(dict):
    """`exec`'s plant is a `setdefault` call, so a subclass override runs."""

    def setdefault(self, key, value=None):
        collect()
        return dict.setdefault(self, key, value)


class ContainsNS(dict):
    """`eval`'s plant asks `__contains__` before it writes."""

    def __contains__(self, key):
        collect()
        return dict.__contains__(self, key)


class SetitemNS(dict):
    """`eval` writes through `__setitem__` when `__builtins__` is absent."""

    def __setitem__(self, key, value):
        collect()
        return dict.__setitem__(self, key, value)


def check(label, fn, expected):
    for round_index in range(60):
        got = fn()
        if got != expected:
            failures.append(f"{label} round {round_index}: {got!r} != {expected!r}")
            return


def exec_setdefault():
    # The code object is compiled here and consumed after the plant.
    namespace = SetdefaultNS()
    namespace["src"] = {"x": 1}
    exec("meth = src.get\nr = meth('x')", namespace)
    return namespace.get("r")


def eval_contains():
    namespace = ContainsNS()
    namespace["src"] = {"x": 1}
    return eval("src.get('x')", namespace)


def eval_setitem():
    namespace = SetitemNS()
    namespace["src"] = {"x": 2}
    return eval("src.get('x')", namespace)


def exec_separate_locals():
    # `locals_arg` is a distinct mapping, so it rides the window too.
    globals_ns = SetdefaultNS()
    locals_ns = {"src": {"x": 3}}
    exec("r = src.get('x')", globals_ns, locals_ns)
    return locals_ns.get("r")


def exec_precompiled_code():
    # A code object the caller already holds must answer the same way.
    namespace = SetdefaultNS()
    namespace["src"] = {"x": 4}
    exec(PRECOMPILED, namespace)
    return namespace.get("r")


PRECOMPILED = compile("r = src.get('x')", "<regression>", "exec")


def warm(rounds):
    """The hot loop the JIT compiles; each round re-enters the window."""
    total = 0
    for _ in range(rounds):
        namespace = SetdefaultNS()
        namespace["src"] = {"x": 1}
        exec("r = src.get('x')", namespace)
        total += namespace.get("r") or 0
    return total


def main():
    check("exec-setdefault", exec_setdefault, 1)
    check("eval-contains", eval_contains, 1)
    check("eval-setitem", eval_setitem, 2)
    check("exec-separate-locals", exec_separate_locals, 3)
    check("exec-precompiled-code", exec_precompiled_code, 4)

    warmed = warm(200)
    if warmed != 200:
        failures.append(f"warm loop: {warmed} != 200")

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS exec namespace code-object rooting")
    return 0


raise SystemExit(main())
