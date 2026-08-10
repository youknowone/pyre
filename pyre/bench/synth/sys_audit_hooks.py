# `sys.addaudithook` / `sys.audit`, the hook set that every interpreter-level
# audit event is delivered to.
#
# The surface pinned here is the part a caller can observe without a hook that
# inspects its own arguments:
#
#   * a hook receives `(event, args)` with `args` as a TUPLE, not the loose
#     arguments -- the emitter's `args_w` list is wrapped once;
#   * hooks fire in installation order;
#   * `sys.audit` is free when no hook is installed, so the count a hook sees
#     starts at the first `addaudithook`, not at interpreter start;
#   * installing a hook emits `sys.addaudithook` to the hooks already there,
#     and an `Exception` out of that event means those hooks REFUSED the new
#     one: it is dropped and the refusal does not propagate.  A `BaseException`
#     outside `Exception` does propagate -- not exercised here, and it cannot
#     be: a hook is installed for the life of the interpreter and the first one
#     that raises masks every hook behind it, so reaching the propagating branch
#     needs the hook set cleared between cases.  Upstream's own facility for
#     that (`__pypy__._testing_clear_audithooks`, `interp_magic.py:292`) refuses
#     to run once translated, so no app-level program on a built interpreter
#     can get there.  The refusal below raises from app code, so it carries a
#     real exception object and takes `error_is_exception`'s isinstance arm;
#     the `PyErrorKind` fallback behind it answers only for an error that never
#     materialised one, which nothing app-level can hand this event.
#   * `__cantrace__` on a hook is honoured (the flag exists so a tracing hook
#     can opt back in); nothing here can observe the tracing state, so only the
#     attribute lookup path is exercised.
#
# The emit loop is hot so the event goes out from compiled code as well as from
# the interpreter; the count is what pins that every iteration emitted exactly
# once.
import sys

N = 20000

seen = []


def hook(event, args):
    if event.startswith("t."):
        seen.append((event, args))


def second(event, args):
    if event == "t.pair":
        seen.append(("second", args))


sys.addaudithook(hook)
sys.addaudithook(second)

# Installation order, and the tuple wrapping.
sys.audit("t.pair", 1, "x")


def emit(n):
    i = 0
    while i < n:
        sys.audit("t.tick", i)
        i = i + 1
    return i


emitted = emit(N)
ticks = sum(1 for event, _ in seen if event == "t.tick")


# A hook that refuses further installs, and one that must therefore never land.
def refuser(event, args):
    if event == "sys.addaudithook":
        raise RuntimeError("refused")


sys.addaudithook(refuser)


def never(event, args):
    seen.append(("NEVER", args))


sys.addaudithook(never)
sys.audit("t.after")

cantraceable = 0


def cantrace_hook(event, args):
    global cantraceable
    if event == "t.cantrace":
        cantraceable = cantraceable + 1


cantrace_hook.__cantrace__ = True
# `refuser` is installed, so this one is refused too -- the count stays 0 and
# the refusal is silent.
sys.addaudithook(cantrace_hook)
sys.audit("t.cantrace")

print(
    [e for e, _ in seen[:3]],
    seen[0][1],
    emitted,
    ticks,
    [e for e, _ in seen[-1:]],
    cantraceable,
)
