# pyre-check: platforms=linux,darwin
# pyre-check: pypy-diverges: `app_main.py` imports the blocked `sys` name, prints an OperationError and exits 1 without opening the prompt
# CPython-suite gap: import tests cover the `None` sentinel and command-line
# tests cover `-i`, but never combine them.
# parity-tests reason: the launcher reads `sys.ps1`/`ps2` from
# `PyInterpreterState` through `_PySys_GetOptionalAttr`; a program's blocked
# `sys.modules['sys']` entry must not prevent the requested prompt.
import subprocess
import sys

PROGRAM = "import sys; sys.modules['sys'] = None"
MARKER = "prompt-ran"


def run(args, feed=""):
    proc = subprocess.run(
        [sys.executable, *args], input=feed, capture_output=True, text=True
    )
    return proc.returncode, proc.stdout


# The control: the assignment itself is ordinary, and a run without `-i` ends
# cleanly, so the rows below are about the prompt and nothing else.
assert run(["-c", PROGRAM]) == (0, "")

# The prompt opens over it and runs what is typed there.
status, out = run(["-i", "-c", PROGRAM], f"print({MARKER!r})\n")
assert status == 0, f"the prompt did not end cleanly: {status}"
assert MARKER in out, f"the prompt did not run the statement it was given: {out!r}"

# And it is the prompt's own status that ends the run, not the program's.
status, _ = run(["-i", "-c", PROGRAM], "raise SystemExit(3)\n")
assert status == 3, f"the prompt's exit status did not reach the shell: {status}"

print("OK")
