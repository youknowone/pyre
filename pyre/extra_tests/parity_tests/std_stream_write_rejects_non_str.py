# CPython-suite gap: test_sys does not import, so nothing in the suite calls
# `sys.stdout.write` with an argument that is not a `str`.
# parity-tests reason: pyre's std streams are native objects that encode
# straight to the descriptor rather than `TextIOWrapper` instances, so the
# argument check `W_TextIOWrapper.write_w` performs had no counterpart and a
# non-`str` write silently reported a zero-length count.

# `click._compat._is_binary_writer` decides whether a stream is the binary
# layer by calling `stream.write(b"")` and seeing whether that raises. A stream
# that accepts it is taken for the binary layer and wrapped in click's own text
# layer, after which every `click.echo` goes to a wrapper over the wrong end
# and nothing reaches the terminal.

import sys

for stream in (sys.stdout, sys.stderr):
    for argument in (b"", b"x", 1, None, [], object()):
        try:
            stream.write(argument)
        except TypeError:
            pass
        else:
            raise SystemExit("write accepted %r" % (type(argument).__name__,))

# A `str` still writes, and still reports the number of characters written.
assert sys.stdout.write("") == 0
assert sys.stderr.write("") == 0

print("OK")
