# `sys._getframe` emits the `sys._getframe` audit event (`vm.py:51`), and the
# constant-depth walker fold that answers the call without running the body has
# to answer for it too.
#
# `AuditHolder`'s hook list is quasi-immutable (`vm.py:439 _immutable_fields_ =
# ['hooks_w?[:]']`), so the fold bakes "no hook installed" behind a
# `guard_not_invalidated` rather than re-reading it, and `addaudithook` revokes
# every loop that baked it.
#
# The hook is installed from INSIDE the loop, halfway through: by then the loop
# is compiled with the event elided, so the install has to revoke it in flight.
# Installing between two calls would not test that -- a fresh call re-traces
# anyway, and the event comes back whether or not the fold depends on the hook
# set.  The tail must therefore deliver one event per remaining iteration.
#
# The hook filters on `f_code.co_name` so an unrelated `_getframe` elsewhere in
# the interpreter cannot inflate the count.
import sys

N = 20000

lines = []


def hook(event, args):
    if event == "sys._getframe" and args[0].f_code.co_name == "loop":
        lines.append(args[0].f_lineno)


def loop(n):
    total = 0
    i = 0
    while i < n:
        f = sys._getframe()
        total = total + f.f_lineno
        if i == n // 2:
            sys.addaudithook(hook)
        i = i + 1
    return total


total = loop(N)

# The frame the hook is handed is the caller's own, read at the
# `sys._getframe()` call -- one line below the `f.f_lineno` line `total` sums,
# so the two are checked against each other rather than against a literal.
print(total == N * (lines[0] + 1), len(lines), len(set(lines)) == 1)
