# CPython-suite gap: `test_io` closes streams it built itself, and
# `test_sys.test_stdout_None` and its neighbours rebind `sys.stdout`, but
# nothing closes a *standard* stream and then uses it -- the one shape where a
# runtime is entitled to shortcuts around the stream object, and therefore the
# one where the checks can go missing.
#
# parity-tests reason: every `_io` method starts at a closed check, so closing
# `sys.stdout` turns the next `print`, `write`, `flush`, `fileno` and
# `writable` into a `ValueError` rather than a write nobody can see or a
# descriptor nobody owns any more.  A runtime that answers these from the
# descriptor keeps going after the stream they stand for is gone, and only a
# standard stream takes that path -- a `TextIOWrapper` the program built itself
# proves nothing about it.
#
# The wording is deliberately not pinned here; which layer's spelling each of
# these carries is `io_closed_message_spelling_is_per_layer.py`.
import subprocess
import sys

PROGRAM = """
import sys

report = sys.stderr


def refusal(label, call):
    try:
        call()
    except ValueError as exc:
        assert "closed file" in str(exc), (label, exc)
        return
    raise AssertionError(label + " answered a closed stream")


sys.stdout.flush()
sys.stdout.close()
assert sys.stdout.closed, "close() left the stream open"
# The whole stack goes, not just the text layer, so a shortcut to the
# descriptor has nothing left to stand in for.
assert sys.stdout.buffer.closed, "buffer stayed open"
assert sys.stdout.buffer.raw.closed, "raw stayed open"

refusal("print", lambda: print("this must not be written"))
refusal("write", lambda: sys.stdout.write("this must not be written\\n"))
refusal("flush", sys.stdout.flush)
refusal("writelines", lambda: sys.stdout.writelines(["nor this\\n"]))
refusal("fileno", sys.stdout.fileno)
refusal("writable", sys.stdout.writable)
# The buffered layer answers the query naming the other direction with a
# constant, without reaching the descriptor, so closing leaves it alone.
assert sys.stdout.readable() is False, "readable() changed"

sys.stdin.close()
refusal("stdin.read", sys.stdin.read)
refusal("stdin.flush", sys.stdin.flush)
refusal("stdin.fileno", sys.stdin.fileno)
refusal("stdin.readable", sys.stdin.readable)
assert sys.stdin.writable() is False, "writable() changed"

print("REFUSED", file=report)
"""

done = subprocess.run(
    [sys.executable, "-c", PROGRAM],
    stdin=subprocess.DEVNULL,
    capture_output=True,
    text=True,
)
assert done.returncode == 0, (done.returncode, done.stderr)
assert done.stdout == "", repr(done.stdout)
assert "REFUSED" in done.stderr, done.stderr

print("OK")
