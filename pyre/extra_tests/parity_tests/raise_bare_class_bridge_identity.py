# CPython-suite gap: exception tests do not alternate bare exception classes at
# one hot raise site after a guard bridge has compiled.
# parity-tests reason: a bridge must keep the normalized exception instance
# distinct from the bare class operand that produced it.

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


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


for exception_class in (ValueError, UserError, RuntimeError):
    probe(exception_class)

print("OK")
