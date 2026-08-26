# pyre-check: gate=1
# `UnicodeError.start` / `.end` are `Py_T_PYSSIZET` members, so
# `PyMember_SetOne` admits only a real `int` for a write.  `object`,
# `encoding` and `reason` are object members and take anything, which is
# what lets `str()` embed a stringified non-str reason.


class MyInt(int):
    pass


class Indexable:
    def __index__(self):
        return 0


def errors(kind):
    if kind == "decode":
        return UnicodeDecodeError("utf-8", b"xy", 0, 1, "r")
    if kind == "encode":
        return UnicodeEncodeError("utf-8", "xy", 0, 1, "r")
    return UnicodeTranslateError("xy", 0, 1, "r")


ACCEPTED = [1, True, MyInt(0)]
# An `__index__` alone is not an `int`, and neither is a float.
REFUSED = ["x", None, 1.0, Indexable()]

for kind in ("decode", "encode", "translate"):
    for attribute in ("start", "end"):
        for value in ACCEPTED:
            exc = errors(kind)
            setattr(exc, attribute, value)
            assert getattr(exc, attribute) == value, (kind, attribute, value)
        for value in REFUSED:
            exc = errors(kind)
            try:
                setattr(exc, attribute, value)
            except TypeError as error:
                assert str(error) == "an integer is required", (
                    kind,
                    attribute,
                    value,
                    str(error),
                )
            else:
                raise AssertionError(f"{kind}.{attribute} = {value!r} was accepted")
            # The refused write left the slot alone.
            assert getattr(exc, attribute) == (0 if attribute == "start" else 1)
            str(exc)

    # The object members take a non-str without complaint.
    for attribute in ("object", "reason") + (() if kind == "translate" else ("encoding",)):
        exc = errors(kind)
        setattr(exc, attribute, 0x345)
        assert getattr(exc, attribute) == 0x345

decoded = errors("decode")
decoded.reason = 0x345
assert str(decoded).endswith(": 837"), str(decoded)
