# CPython-suite gap: test_re never resizes a bytearray from inside re.sub.
# parity-tests reason: subx and split_w hold one buffer export for the whole scan, so a callback that resizes the bytearray is refused and the export is released when the scan returns.
# pyre-check: pypy-diverges: make_ctx releases the export immediately and BufMatchContext reads the live buffer with ctx.end fixed at entry, so pypy3 reads past the shrunk buffer (the oracle prints b'aaYY') instead of raising BufferError.

"""sub holds the subject's buffer export across the replacement callback."""

import re

subject = bytearray(b"aaXaaXaa")


def repl(match):
    del subject[match.end() :]
    return b"Y"


raised = False
try:
    re.sub(br"X", repl, subject)
except BufferError:
    raised = True
assert raised, "re.sub callback resize must raise BufferError"
assert bytes(subject) == b"aaXaaXaa", bytes(subject)
# state_fini released the export: a later resize is allowed.
subject.append(ord(b"Z"))
assert bytes(subject) == b"aaXaaXaaZ", bytes(subject)

# split_w has no callback. The export is still acquired for the scan and
# released before split returns, so the bytearray is resizable afterwards.
split_subject = bytearray(b"aaXaaXaa")
parts = re.split(br"X", split_subject)
assert parts == [b"aa", b"aa", b"aa"], parts
del split_subject[:]
assert bytes(split_subject) == b"", bytes(split_subject)

# A bytes pattern on memoryview(bytearray): the view exports the bytearray,
# and subx holds that export across the callback, so the resize is refused.
view_subject = bytearray(b"aaXaaXaa")
view = memoryview(view_subject)


def repl_view(match):
    del view_subject[match.end() :]
    return b"Y"


view_raised = False
try:
    re.sub(br"X", repl_view, view)
except BufferError:
    view_raised = True
assert view_raised, "re.sub on memoryview(bytearray) must refuse the resize"
assert bytes(view_subject) == b"aaXaaXaa", bytes(view_subject)

print("OK")
