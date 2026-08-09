"""`sys.addaudithook` installs a hook that `sys.audit` actually calls.

`addaudithook` was `|_| Ok(w_none())` — it accepted a hook and dropped it — so
no hook could ever fire and the 21 stdlib modules that call `sys.audit`
(`shutil`, `subprocess`, `webbrowser`, `glob`, `pickle`, ...) reported nothing
to anyone.

A hook can never be removed, so every assertion below has to be written for a
process whose hook set only grows; the raising cases route through one
module-level switch that is put back immediately.  The interpreters also
disagree on which *internal* operations raise an event at all — CPython audits
`compile`, `exec`, `import` and much else, pyre raises only what a
`sys.audit(...)` call names — so the recorder keeps just the events this file
names, plus `sys.addaudithook`, which both raise.
"""

import sys

ERRORS = []


def check(cond, what):
    if not cond:
        ERRORS.append(what)


def raises(what, exc, fn):
    try:
        fn()
    except exc as e:
        return e
    except BaseException as e:
        ERRORS.append(f"{what}: raised {type(e).__name__}({e}), expected {exc.__name__}")
        return None
    ERRORS.append(f"{what}: no exception, expected {exc.__name__}")
    return None


# Only these reach a recorder, so an interpreter that audits more of its own
# internals than the other does not change what is compared.
WATCHED = {"pyre.first", "pyre.second", "pyre.raising", "sys.addaudithook"}

# Set to an exception instance to make every recorder raise it; put back to
# None on the next line.  A hook that raises stops the hooks after it, so this
# is never left set.
raiser = None


class Recorder:
    def __init__(self, name):
        self.name = name
        self.seen = []

    def __call__(self, event, args):
        if event in WATCHED:
            self.seen.append((event, args))
        if raiser is not None:
            raise raiser

    def events(self):
        return [event for event, _ in self.seen]


# ── with no hook installed, audit is still the no-op it was ───────────────
check(sys.audit("pyre.first") is None, "audit with no hook did not return None")


# ── the first hook ────────────────────────────────────────────────────────
first = Recorder("first")
sys.addaudithook(first)
# The `sys.addaudithook` event is raised BEFORE the hook is stored, so a hook
# never sees its own installation.
check(first.seen == [], f"a new hook saw its own installation: {first.seen}")

sys.audit("pyre.first", 1, 2)
check(len(first.seen) == 1, f"hook was not called: {first.seen}")
if first.seen:
    event, args = first.seen[0]
    check(event == "pyre.first", f"event was {event!r}")
    check(args == (1, 2), f"args were {args!r}")
    check(type(args) is tuple, f"args was a {type(args).__name__}, expected tuple")
    # `@unwrap_spec(event="text")` unwraps and re-wraps, so what the hook is
    # handed is a plain `str` even when the caller named a subclass.
    check(type(event) is str, f"event was a {type(event).__name__}, expected str")

# No argument at all is an empty tuple, not None.
first.seen.clear()
sys.audit("pyre.first")
check(first.seen == [("pyre.first", ())], f"no-argument audit gave {first.seen}")


class SubStr(str):
    pass


first.seen.clear()
sys.audit(SubStr("pyre.first"))
check(
    first.seen and type(first.seen[0][0]) is str,
    f"a str subclass event reached the hook as {first.seen}",
)


# ── a second hook, and the order they run in ──────────────────────────────
second = Recorder("second")
first.seen.clear()
sys.addaudithook(second)
check(
    first.events() == ["sys.addaudithook"],
    f"installing a hook did not raise sys.addaudithook: {first.events()}",
)
check(second.seen == [], f"the second hook saw its own installation: {second.seen}")

first.seen.clear()
second.seen.clear()
sys.audit("pyre.second", "x")
check(first.seen == [("pyre.second", ("x",))], f"first hook: {first.seen}")
check(second.seen == [("pyre.second", ("x",))], f"second hook: {second.seen}")


# ── a hook that raises on the install event refuses the new hook ──────────
# The new hook is never stored, whatever was raised.  What the class decides is
# only whether the caller hears about it: anything derived from `Exception` is
# swallowed and `addaudithook` returns None, while a `BaseException` that is
# not an `Exception` comes back out.
def install_under(exc, hook):
    """Install `hook` while every recorder raises `exc`; report what happened."""
    global raiser
    first.seen.clear()
    second.seen.clear()
    hook.seen.clear()
    raiser = exc
    try:
        return ("returned", sys.addaudithook(hook))
    except BaseException as e:  # noqa: BLE001 - the outcome is the measurement
        return ("raised", type(e))
    finally:
        raiser = None


def installed(hook):
    """Whether `hook` is in the hook list, asked by raising an event."""
    hook.seen.clear()
    sys.audit("pyre.second")
    return hook.seen != []


for exc, expected in (
    (RuntimeError("no more hooks"), ("returned", None)),
    (ValueError("not a veto"), ("returned", None)),
    (Exception("plain"), ("returned", None)),
    (KeyboardInterrupt(), ("raised", KeyboardInterrupt)),
):
    name = type(exc).__name__
    refused = Recorder(name)
    outcome = install_under(exc, refused)
    check(outcome == expected, f"addaudithook under {name}: {outcome}, expected {expected}")
    # The refusal came from the FIRST hook, so the second one never ran.
    check(first.events() == ["sys.addaudithook"], f"under {name}, first hook: {first.events()}")
    check(second.events() == [], f"under {name}, the refusal did not stop hook 2")
    check(not installed(refused), f"a hook refused under {name} was installed anyway")

# The hooks that were already there are untouched by any of that.
first.seen.clear()
second.seen.clear()
sys.audit("pyre.second")
check(len(first.seen) == 1 and len(second.seen) == 1, "the surviving hooks stopped firing")


# ── an exception from a hook during an ordinary audit propagates ──────────
first.seen.clear()
second.seen.clear()
raiser = ValueError("from a hook")
try:
    raises("audit with a raising hook", ValueError, lambda: sys.audit("pyre.raising"))
finally:
    raiser = None
check(first.events() == ["pyre.raising"], f"first hook: {first.events()}")
check(
    second.events() == [],
    f"a raising hook did not stop the ones after it: {second.events()}",
)


# ── and the hooks still work afterwards ───────────────────────────────────
first.seen.clear()
second.seen.clear()
check(sys.audit("pyre.second", 7) is None, "audit stopped returning None")
check(first.seen == [("pyre.second", (7,))], f"first hook: {first.seen}")
check(second.seen == [("pyre.second", (7,))], f"second hook: {second.seen}")

# The argument checks in front of the dispatch still report, with hooks
# installed, before any hook is reached.
first.seen.clear()
raises("audit(123) with hooks installed", TypeError, lambda: sys.audit(123))
check(first.seen == [], f"a bad event name reached the hooks: {first.seen}")

if ERRORS:
    for e in ERRORS:
        sys.stderr.write(f"FAIL: {e}\n")
    raise AssertionError(f"{len(ERRORS)} divergence(s)")

print("OK")
