"""`os.dup2` binds its third argument by name, and honours it.

`dup2(fd, fd2, inheritable=False)` is the call that asks for a descriptor an
`exec` will not carry.  A registration that cannot split keywords off the
argument slice drops the request silently — the answer is still a descriptor,
so nothing raises and only `get_inheritable` tells you the flag went the other
way.  That is what this asserts first; the binding errors around it are the
same defect seen from the side where it is loud.

The arity *message* text is deliberately not asserted.  CPython spells it two
ways depending on whether the parameters sit after the positional-only `/`, and
pyre's builtins do not yet carry that marker — only the exception type is
parity today.
"""

import os


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def raises(what, fn):
    try:
        fn()
    except TypeError:
        return
    except Exception as e:
        raise AssertionError(f"{what}: raised {type(e).__name__}, expected TypeError")
    raise AssertionError(f"{what}: no exception")


r, w = os.pipe()
# A descriptor we own and may overwrite. dup2 closes its target first, so
# every case below can reuse this number.
target = os.dup(r)
try:
    # ── the property ─────────────────────────────────────────────────────
    os.dup2(r, target, inheritable=False)
    check(
        os.get_inheritable(target) is False,
        "dup2(..., inheritable=False) produced an inheritable descriptor",
    )
    # The same request positionally, so a failure tells keyword binding from
    # the flag being ignored outright.
    os.dup2(r, target, False)
    check(
        os.get_inheritable(target) is False,
        "dup2(..., False) produced an inheritable descriptor",
    )
    # The default is the other way, so the assertion above cannot pass by a
    # constant answer.
    os.dup2(r, target)
    check(
        os.get_inheritable(target) is True,
        "dup2 defaulted to a non-inheritable descriptor",
    )
    os.dup2(r, target, True)
    check(os.get_inheritable(target) is True, "dup2(..., True) was not inheritable")

    # ── the binding, from the side that raises ───────────────────────────
    # Both required arguments by name.
    check(os.dup2(fd=r, fd2=target) == target, "dup2(fd=, fd2=) did not answer fd2")
    # One positional, one by name.
    check(os.dup2(r, fd2=target) == target, "dup2(r, fd2=) did not answer fd2")

    raises("dup2 with an unknown keyword", lambda: os.dup2(r, target, zzz=1))
    raises("dup2 with 5 positionals", lambda: os.dup2(r, target, True, "x", "y"))
    raises("dup2 with fd2 given twice", lambda: os.dup2(r, target, fd2=target))
    raises("dup2 with one argument", lambda: os.dup2(r))
    raises("dup2 with no arguments", lambda: os.dup2())
    raises("dup2 with a str fd", lambda: os.dup2("a", target))
finally:
    for fd in (target, r, w):
        try:
            os.close(fd)
        except OSError:
            pass

print("OK")
