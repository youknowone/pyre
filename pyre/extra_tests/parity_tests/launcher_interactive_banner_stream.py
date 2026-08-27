# pyre-check: platforms=linux,darwin
# CPython-suite gap: the two modules that could see this both sit at
# IMPORTERROR in the baseline, so neither runs an assertion today.  Even
# restored they would not catch it: `test_repl` reaches the prompt only through
# `spawn_repl`, whose signature is `stderr=subprocess.STDOUT` and which no call
# site overrides, so every assertion there reads the two streams merged; and
# `test_cmd_line` never spawns a prompt at all -- its children take a pipe for
# stdin, which suppresses the banner outright.
#
# parity-tests reason: `pymain_header` writes both of its lines with
# `fprintf(stderr, ...)`, and `app_main.py print_banner` writes them through
# `sys.stderr`, so the two references agree that the prompt's greeting is not
# part of its stdout.  What that buys is a session whose stdout can be captured
# on its own -- a pipe, a doctest harness, an editor's REPL pane -- and a
# runtime that prints the banner on stdout instead corrupts the first thing
# such a caller reads while passing every test that merges the streams.
#
# The prompt only prints a banner when stdin is a terminal, so the child is
# driven through a pty; that is what the `platforms` header is for.
import os
import pty
import re
import select
import sys
import tempfile

BANNER = re.compile(rb"^(Python 3|pyre )", re.M)


def banner_streams(argv):
    """`(banner lines on stdout, banner lines on stderr)` for an interactive run."""
    # stderr is the only stream that can be redirected away from the terminal
    # without making stdin a pipe, which would suppress the banner outright.
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".err")
    handle.close()
    pid, fd = pty.fork()
    if pid == 0:  # child: stdin and stdout are the terminal, stderr is the file
        try:
            err = os.open(handle.name, os.O_WRONLY | os.O_TRUNC)
            os.dup2(err, 2)
            os.execv(argv[0], argv)
        finally:
            os._exit(127)
    os.write(fd, b"exit()\n")
    out = b""
    while True:
        readable, _, _ = select.select([fd], [], [], 30.0)
        if not readable:
            break
        try:
            chunk = os.read(fd, 4096)
        except OSError:  # the child closed the terminal
            break
        if not chunk:
            break
        out += chunk
    os.waitpid(pid, 0)
    with open(handle.name, "rb") as stream:
        err_text = stream.read()
    os.unlink(handle.name)
    return len(BANNER.findall(out)), len(BANNER.findall(err_text))


stdout_lines, stderr_lines = banner_streams([sys.executable])
# The terminal echoes what was typed into it, so stdout carries the `exit()`
# that drove the session; what it must not carry is the banner.
assert stdout_lines == 0, (stdout_lines, stderr_lines)
assert stderr_lines == 1, (stdout_lines, stderr_lines)

# `-q` drops it from stderr too, rather than moving it.
quiet_out, quiet_err = banner_streams([sys.executable, "-q"])
assert (quiet_out, quiet_err) == (0, 0), (quiet_out, quiet_err)

print("OK")
