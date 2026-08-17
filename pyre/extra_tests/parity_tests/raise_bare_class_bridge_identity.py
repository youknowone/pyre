# CPython-suite gap: exception tests do not alternate bare exception classes at
# one hot raise site after a guard bridge has compiled.
# parity-tests reason: a bridge must keep the normalized exception instance
# distinct from the bare class operand that produced it.

# No `pypyjit.set_param` preamble on purpose.  The shape needs three traces in
# sequence — a loop that folds the bare-class raise, a second loop that declines
# it to the normalizing residual, then a bridge off that residual's class guard.
# Lowering the thresholds compiles the first loop before the sequence can form,
# and the fixture then passes on a binary that carries the defect.  The default
# thresholds reach every trace well inside `N`.

N = 200000


class UserError(Exception):
    pass


def raise_it(cls):
    raise cls


def probe(cls):
    caught = 0
    last = None
    i = 0
    while i < N:
        try:
            raise_it(cls)
        except BaseException as exc:
            caught += 1
            last = exc
        i += 1
    assert caught == N, (cls, caught)
    assert type(last) is cls, (cls, type(last))
    return last


# Order is the condition: a class the bare-class raise fold accepts, then one it
# declines, then a second accepted class not yet seen at this site.
for exception_class in (ValueError, UserError, RuntimeError):
    probe(exception_class)

print("OK")
