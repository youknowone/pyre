# CPython-suite gap: traceback mutation tests do not exercise JIT warmup with
# alternating retained, shortened, and cleared tracebacks through a finally.
# parity-tests reason: exception propagation must preserve carrier state across
# compiled execution and handler resumption, including same-frame reraises.

"""Reraising preserves exception identity, context, and an edited traceback."""


def origin(exc):
    raise exc


def run(n):
    total = 0
    i = 0
    while i < n:
        context = ValueError("context")
        original = KeyError("original")
        try:
            raise context
        except ValueError:
            try:
                try:
                    origin(original)
                except KeyError as caught:
                    assert caught is original
                    assert caught.__context__ is context
                    tb = caught.__traceback__
                    assert tb.tb_frame.f_code.co_name == "run"
                    assert tb.tb_next.tb_frame.f_code.co_name == "origin"
                    assert tb.tb_next.tb_next is None
                    if i % 3 == 0:
                        expected = None
                    elif i % 3 == 1:
                        expected = tb.tb_next
                    else:
                        expected = tb
                    caught.__traceback__ = expected
                    try:
                        raise
                    finally:
                        total += 1
            except KeyError as reraised:
                assert reraised is original
                assert reraised.__context__ is context
                assert reraised.__traceback__ is expected, (i, i % 3)
                assert reraised.__cause__ is None
                assert not reraised.__suppress_context__
                total += 1
        i += 1
    return total


assert run(6) == 12
assert run(6000) == 12000
print("OK")
