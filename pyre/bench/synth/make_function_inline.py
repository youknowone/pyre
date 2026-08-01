# `MAKE_FUNCTION` reaches `NewWithVtable` + `SetfieldGc` instead of the
# `jit_make_function_from_globals` residual.
#
# No `max-pypy-ratio`: this fixture exists to pin what the emit BUILDS.
# `call_loop_local_function` carries the fold's perf gate.
#
# `pyopcode.py:1457 MAKE_FUNCTION` runs `function.py:47-57 Function.__init__`
# per execution, so a `def` in a loop body allocates a function per iteration.
# Emitting that construction inline lets the optimizer drop it when the
# function never escapes; when it DOES escape, the materialized object must be
# indistinguishable from the interpreted one.  `hot` is the escaping half — it
# reads back every slot the emit writes and every slot it leaves null, on a
# function built inside the hot loop, so a wrong constant or a missing store
# shows up as a changed checksum rather than a silent divergence.
#
# The nursery is not zero-filled, so a pointer slot missing from the size
# descr's census would keep recycled bytes: `probe_nulls` reads each one that
# `Function.__init__` leaves unset.  `can_change_code` is a plain byte with no
# NULL store behind the allocation, which `__code__` assignment checks.
#
# `N` only has to carry `hot` past the trace threshold: every read in its body
# is its own residual, so raising it buys no coverage and only costs run time
# in a suite this fixture is not a perf member of.
N = 20000


def probe_nulls(f):
    # Every slot `Function.__init__` leaves unset, read through its app-level
    # surface.  A garbage pointer here is a wild dereference, not a wrong
    # answer, so reaching the end at all is most of the assertion.
    out = 0
    out += len(f.__dict__)
    out += 1 if f.__defaults__ is None else 0
    out += 1 if f.__kwdefaults__ is None else 0
    out += 1 if f.__closure__ is None else 0
    out += 1 if f.__doc__ is None else 0
    out += len(f.__annotations__)
    out += 1 if f.__module__ == __name__ else 0
    return out


def hot():
    acc = 0
    i = 0
    while i < N:

        def grab(a=None):
            # `len` is absent from this module's globals, so resolving it walks
            # the builtin mapping the constructor froze onto the function.  A
            # wrong or missing one raises NameError instead of answering 0.
            return len(())

        # The slots the emit writes.
        acc += grab()
        acc += len(grab.__name__)
        acc += len(grab.__qualname__)
        acc += 1 if grab.__globals__ is globals() else 0
        acc += 1 if grab.__code__ is grab.__code__ else 0
        # `can_change_code` must be True: a byte left at whatever the recycled
        # nursery held would raise here roughly half the time.
        grab.__code__ = grab.__code__
        acc += probe_nulls(grab)
        i += 1
    return acc


def escapes():
    # The same `def`, kept alive past the loop, so the virtual is forced and
    # materialized rather than dropped.
    fns = []
    i = 0
    while i < 200:

        def grab():
            return i

        fns.append(grab)
        i += 1
    return len(fns), fns[0].__qualname__, fns[0] is not fns[1]


def with_attributes():
    # SET_FUNCTION_ATTRIBUTE writes land on the emitted object, so defaults and
    # the closure tuple must survive the same construction.
    out = []
    i = 0
    while i < 200:
        cell = i

        def grab(a, b=7, *, c=9):
            return a + b + c + cell

        out.append(grab(1))
        i += 1
    return out[-1], grab.__defaults__, grab.__kwdefaults__, len(grab.__closure__)


def foreign_globals():
    # `exec` with a bare dict: `__builtins__` is inserted as the builtin MODULE
    # here, while a mapping that is not a module declines the fold and keeps
    # the residual.  Both must build the same function.
    src = "def made():\n    return 5\n"
    plain = {}
    exec(src, plain)
    mapped = {"__builtins__": {"len": len}}
    exec(src, mapped)
    return (
        plain["made"](),
        plain["made"].__qualname__,
        mapped["made"](),
        plain["made"].__globals__ is plain,
    )


def main():
    print(hot())
    print(escapes())
    print(with_attributes())
    print(foreign_globals())
    # The code object owns one realized `co_qualname`, so repeated reads and
    # every function built from it name the same object.
    c = hot.__code__
    print(c.co_qualname is c.co_qualname, hot.__qualname__ == c.co_qualname)


main()
