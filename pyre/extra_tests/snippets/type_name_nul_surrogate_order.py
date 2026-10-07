# pyre-check: gate=1
# A type name that contains both U+0000 and a lone surrogate is rejected as
# UnicodeEncodeError from the utf8 scan, not as ValueError from the NUL
# test.  type() and the __name__ setter share that order.


def expect_encode(fn):
    try:
        fn()
    except UnicodeEncodeError as error:
        assert error.reason == "surrogates not allowed", error.reason
        return error
    raise AssertionError("expected UnicodeEncodeError")


error = expect_encode(lambda: type("x\x00\ud800", (), {}))
assert error.start == 2 and error.end == 3, (error.start, error.end)

error = expect_encode(lambda: type("x\ud800\x00", (), {}))
assert error.start == 1 and error.end == 2, (error.start, error.end)


class T:
    pass


error = expect_encode(lambda: setattr(T, "__name__", "x\x00\ud800"))
assert error.start == 2 and error.end == 3, (error.start, error.end)
assert T.__name__ == "T"

try:
    type("x\x00y", (), {})
except ValueError as error:
    assert "null characters" in str(error), str(error)
else:
    raise AssertionError("expected ValueError")
