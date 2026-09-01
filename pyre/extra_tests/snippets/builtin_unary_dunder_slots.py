# pyre-check: gate=1
# An unbound numeric unary slot performs the arithmetic; it does not look the
# dunder up again.  `descroperation.py` builds `int.__neg__` from
# `W_IntObject.descr_neg`, the arithmetic alone -- the override lookup lives in
# `_make_unaryop_impl`, which is what the `-x` operator runs, not the slot.


class NegInt(int):
    def __neg__(self):
        return "custom neg"


class PosInt(int):
    def __pos__(self):
        return "custom pos"


class InvInt(int):
    def __invert__(self):
        return "custom inv"


# The operator still dispatches to the override.
assert -NegInt(5) == "custom neg"
assert +PosInt(5) == "custom pos"
assert ~InvInt(5) == "custom inv"

# The unbound slot never does.
assert int.__neg__(NegInt(5)) == -5
assert int.__pos__(PosInt(5)) == 5
assert type(int.__pos__(PosInt(5))) is int
assert int.__invert__(InvInt(5)) == -6


class NegFloat(float):
    def __neg__(self):
        return "custom neg"


class NegComplex(complex):
    def __neg__(self):
        return "custom neg"


assert -NegFloat(1.5) == "custom neg"
assert float.__neg__(NegFloat(1.5)) == -1.5
assert -NegComplex(1 + 2j) == "custom neg"
assert complex.__neg__(NegComplex(1 + 2j)) == complex(-1, -2)


# An override delegating back to the builtin slot terminates.
class DelegatesNeg(int):
    def __neg__(self):
        return int.__neg__(self)


class DelegatesPos(int):
    def __pos__(self):
        return int.__pos__(self)


class DelegatesInv(int):
    def __invert__(self):
        return int.__invert__(self)


assert -DelegatesNeg(5) == -5
assert +DelegatesPos(5) == 5
assert ~DelegatesInv(5) == -6

# A receiver that is not an instance of the slot's own type is rejected by the
# descriptor, before the structural body could reach `try_instance_unaryop` and
# recurse back through the override.
for base, op in ((int, "__neg__"), (int, "__pos__"), (int, "__invert__"),
                 (float, "__neg__"), (complex, "__neg__")):
    namespace = {}
    exec(
        "class Bad:\n"
        f"    def {op}(self): return {base.__name__}.{op}(self)\n",
        namespace,
    )
    try:
        getattr(namespace["Bad"](), op)()
    except TypeError as error:
        assert str(error) == (
            f"descriptor '{op}' requires a '{base.__name__}' object "
            "but received a 'Bad'"
        ), error
    else:
        raise AssertionError(f"{base.__name__}.{op} accepted a foreign receiver")

# The slots reject a missing operand rather than indexing an empty argument
# list.
for slot in (int.__neg__, int.__pos__, int.__invert__):
    try:
        slot()
    except TypeError:
        pass
    else:
        raise AssertionError("no TypeError for a missing operand")

# bool keeps int's slots by inheritance.
assert int.__neg__(True) == -1
assert int.__pos__(True) == 1
