# pyre-check: platforms=linux,darwin
#
# CPython-suite gap: `test_cmd_line.test_inspect_via_env_var` is the only test
# for the variable, it pipes the program rather than giving it a terminal, and
# `test_cmd_line` sits at IMPORTERROR in the baseline anyway.  Nothing anywhere
# sets the name from inside the program, which is the case the comment in
# `pymain_repl` exists for.
#
# parity-tests reason: "Check this environment variable at the end, to give
# programs the opportunity to set it from Python."  `os.environ` writes reach
# `putenv`, so a program that sets PYTHONINSPECT during its own run is asking
# for the prompt that follows it -- and a launcher that folded the variable into
# its option block at startup cannot see the request.  The terminal is part of
# the case: `pymain_repl` needs `stdin_is_interactive`, which the variable does
# not supply.
import os
import pty
import select
import sys
import time

PROGRAM = """
import os
os.environ["PYTHONINSPECT"] = "1"
print("program ran")
"""


def under_a_terminal(args, feed):
    """`args` run with a pty on stdin, driven by `feed` and then EOF."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, *args])
    out = b""
    os.write(fd, feed.encode())
    deadline = time.time() + 20.0
    quiet = 0
    while time.time() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.5)
        if readable:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
            quiet = 0
            continue
        quiet += 1
        if quiet == 1:
            os.write(fd, b"\x04")
        elif quiet > 4:
            break
    try:
        os.close(fd)
    except OSError:
        pass
    os.waitpid(pid, 0)
    return out.decode("utf-8", "replace")


out = under_a_terminal(["-c", PROGRAM], 'print("PROMPT REACHED")\n')
assert "program ran" in out, out
assert "PROMPT REACHED" in out, out

print("OK")
