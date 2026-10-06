# CPython-suite gap: exception tests omit a hot body entered while a caller
# handles E, that catches a bare reraise and then explicitly raises the same E
# from a new exception K.
# parity-tests reason: a same-walk bare reraise must not suppress the later
# explicit raise's `__context__` store for that same instance.

"""A later explicit raise chains __context__ after a same-walk bare reraise."""

ROUNDS = 20000


def body(inner):
    try:
        raise
    except KeyError as exc:
        try:
            raise inner
        except IndexError:
            raise exc


def count_callee():
    """Each iteration calls a callee that starts while the caller handles E."""
    lost = 0
    for _ in range(ROUNDS):
        outer = KeyError("outer")
        inner = IndexError("inner")
        try:
            raise outer
        except KeyError:
            try:
                body(inner)
            except KeyError as caught:
                if caught is not outer or caught.__context__ is not inner:
                    lost += 1
    return lost


def count_same_frame():
    """Loop body entered while this frame handles a fresh E."""
    lost = 0
    for _ in range(ROUNDS):
        outer = KeyError("outer")
        inner = IndexError("inner")
        try:
            raise outer
        except KeyError as exc:
            try:
                try:
                    raise
                except KeyError:
                    try:
                        raise inner
                    except IndexError:
                        raise exc
            except KeyError as caught:
                if caught is not exc or caught.__context__ is not inner:
                    lost += 1
    return lost


def count_bare_reraise_untouched():
    """A bare reraise must leave an already-stamped __context__ untouched."""
    outer = KeyError("outer")
    marker = ValueError("marker")
    outer.__context__ = marker
    lost = 0
    try:
        raise outer
    except KeyError:
        for _ in range(ROUNDS):
            try:
                raise
            except KeyError as caught:
                if caught is not outer or caught.__context__ is not marker:
                    lost += 1
    return lost


callee = count_callee()
same_frame = count_same_frame()
untouched = count_bare_reraise_untouched()

assert callee == 0, f"callee lost {callee}/{ROUNDS}"
assert same_frame == 0, f"same-frame lost {same_frame}/{ROUNDS}"
assert untouched == 0, f"bare reraise mutated context {untouched}/{ROUNDS}"

print("OK")
