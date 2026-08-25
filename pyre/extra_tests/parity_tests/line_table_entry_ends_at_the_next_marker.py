# CPython-suite gap: every `co_linetable` the suite reads was written by the
# compiler, so each entry's payload ends exactly where the next entry's header
# begins and the two ways of finding that boundary agree.  `code.replace(
# co_linetable=...)` stores arbitrary bytes, and nothing in the suite does.
#
# parity-tests reason: the two walks over the table are different functions
# upstream and stop at different places on purpose.  `advance`, which
# `co_lines()` and every line resolution use, reads the line delta *without*
# moving the cursor and then skips forward to the next byte with bit 7 set --
# so an entry ends at the next marker, not after however many payload bytes its
# code declares.  `advance_with_locations`, which `co_positions()` uses, really
# does consume the payload.  A single byte can be a payload by length and a
# header by its marker, and only the marker decides for the first walk.
#
# PyPy 7.3.20 is a 3.11 line table and rejects a `co_linetable` this short.

TABLES = [
    b"",
    b"\x00",
    b"\x80",
    # `\x00` declares a one-byte `Short0` payload and `\x80` is that byte by
    # length -- but it carries the marker, so it opens a second range instead.
    b"\x00\x80",
    # Code 15 without the marker is a location kind, not a no-location range:
    # `co_lines()` reports the computed line and `co_positions()` reports None.
    b"\x78",
    b"\xf8",
    b"\xf8\xf8",
    # A `NoColumns` entry whose signed varint carries the line far negative.
    b"\xe8\x7f",
    # A header claiming eight code units, followed by the same two bytes.
    b"\x0f\x00\x80",
]


# Compiled from a string so `co_firstlineno` is 1 and the lines printed below
# do not move when this file is edited.
NAMESPACE = {}
exec(compile("def probe():\n    return 1\n", "<table>", "exec"), NAMESPACE)
BASE = NAMESPACE["probe"].__code__

for table in TABLES:
    code = BASE.replace(co_linetable=table)
    print(repr(table))
    print("   co_lines:", list(code.co_lines()))
    print("   co_positions:", list(code.co_positions()))

print("OK")
