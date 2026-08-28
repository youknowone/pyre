# pyre-check: pypy-diverges: pypy3's `_io` raises the `fileio.c` spelling for
# every layer, so none of its messages carry the trailing period.  The wording
# is what this pins, so the divergence fails the fixture rather than being
# expressible in it.
#
# CPython-suite gap: `test_io` closes streams and asserts that `ValueError` is
# raised, never what it says, so the two spellings are interchangeable to every
# test in the tree.
#
# parity-tests reason: `_io` carries two literals on purpose -- `textio.c` and
# `iobase.c` end the sentence with a period and `fileio.c` and `stringio.c` do
# not -- and which one a program sees says which layer refused it.  A runtime
# that picks one spelling for all of them answers the wrong layer.
import io
import os

PERIOD = "I/O operation on closed file."
NO_PERIOD = "I/O operation on closed file"


def refusal(call):
    try:
        call()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("a closed stream accepted the call")


text = io.TextIOWrapper(io.BufferedWriter(io.FileIO(os.open(os.devnull, os.O_WRONLY), "w")))
raw = io.FileIO(os.open(os.devnull, os.O_WRONLY), "w")
string = io.StringIO()
byte = io.BytesIO()
base = io.IOBase()
for stream in (text, raw, string, byte, base):
    stream.close()

# `textio.c CHECK_CLOSED` and the four `iobase.c` sites reached through a bare
# `_IOBase`, none of which a concrete class overrides.
assert refusal(lambda: text.write("x")) == PERIOD, refusal(lambda: text.write("x"))
assert refusal(base.flush) == PERIOD, refusal(base.flush)
assert refusal(base.isatty) == PERIOD, refusal(base.isatty)
assert refusal(base._checkClosed) == PERIOD, refusal(base._checkClosed)
assert refusal(lambda: base.writelines([])) == PERIOD, refusal(lambda: base.writelines([]))

# `fileio.c` and `stringio.c` state it without one.
assert refusal(lambda: raw.write(b"x")) == NO_PERIOD, refusal(lambda: raw.write(b"x"))
assert refusal(lambda: string.write("x")) == NO_PERIOD, refusal(lambda: string.write("x"))

# `bytesio.c` sides with the period, so the split is not raw-versus-text.
assert refusal(lambda: byte.write(b"x")) == PERIOD, refusal(lambda: byte.write(b"x"))

print("OK")
