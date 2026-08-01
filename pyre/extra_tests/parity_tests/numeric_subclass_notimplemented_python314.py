import operator


binary_operations = (
    ("__add__", "__radd__", operator.add),
    ("__sub__", "__rsub__", operator.sub),
    ("__mul__", "__rmul__", operator.mul),
    ("__truediv__", "__rtruediv__", operator.truediv),
    ("__floordiv__", "__rfloordiv__", operator.floordiv),
    ("__mod__", "__rmod__", operator.mod),
    ("__pow__", "__rpow__", operator.pow),
    ("__lshift__", "__rlshift__", operator.lshift),
    ("__rshift__", "__rrshift__", operator.rshift),
    ("__and__", "__rand__", operator.and_),
    ("__or__", "__ror__", operator.or_),
    ("__xor__", "__rxor__", operator.xor),
    ("__divmod__", "__rdivmod__", divmod),
)

for forward_name, reverse_name, operation in binary_operations:
    calls = []

    def forward(self, other, name=forward_name):
        calls.append(name)
        return NotImplemented

    def reverse(self, other, name=reverse_name):
        calls.append(name)
        return NotImplemented

    Number = type(
        "Number",
        (int,),
        {forward_name: forward, reverse_name: reverse},
    )
    try:
        operation(Number(7), Number(3))
    except TypeError:
        pass
    else:
        raise AssertionError((forward_name, "TypeError not raised"))
    # Same-type binary dispatch tries the forward slot exactly once and does
    # not retry it before falling back to the builtin storage operation.
    assert calls == [forward_name], (forward_name, calls)


class RepeatCount(int):
    def __mul__(self, other):
        return NotImplemented

    def __rmul__(self, other):
        return NotImplemented


assert RepeatCount(2) * [1] == [1, 1]
assert [1] * RepeatCount(2) == [1, 1]


for base, value in ((float, 1.5), (complex, 1 + 2j)):
    Number = type(
        "Number",
        (base,),
        {
            "__add__": lambda self, other: NotImplemented,
            "__radd__": lambda self, other: NotImplemented,
        },
    )
    try:
        Number(value) + Number(value)
    except TypeError:
        pass
    else:
        raise AssertionError((base, "TypeError not raised"))

print("OK")
