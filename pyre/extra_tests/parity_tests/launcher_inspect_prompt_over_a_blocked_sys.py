# pyre-check: platforms=linux,darwin
# pyre-check: pypy-diverges: `app_main.py` reaches its prompt through an
# app-level `import sys`, so the sentinel the program left behind reaches the
# launcher itself: pypy3 leaves `run_command_line` through an unhandled
# `ModuleNotFoundError`, prints an interpreter-level `debug: OperationError`
# dump and exits 1 without opening the prompt.  Its control row -- the same
# program without `-i` -- agrees at 0, which places the divergence in the
# prompt rather than in the assignment.
#
# CPython-suite gap: the suite pins the `None` sentinel as an import failure
# (`test_importlib.import_.test_api` `test_blocked_fromlist`) and it pairs `-i`
# with a program exactly once (`test_cmd_line.test_run_module_bug1764407`,
# which asserts a name, not what the prompt could still reach).  Nothing puts
# the two together, so nothing covers whether a program can block the prompt
# that follows it.
#
# parity-tests reason: `sys.modules["name"] = None` is the documented way to
# stop everything downstream from importing that name, and code reaches for it
# on `sys` as readily as on anything else.  The prompt `-i` opens is not
# downstream of the program in that sense: it belongs to the launcher, and
# `_PySys_GetOptionalAttr` reads `sys.ps1`/`ps2` off `PyInterpreterState`'s own
# `sys` rather than through the mapping the program just rebound.  So the
# assignment is the program's to make and the prompt still opens over it -- a
# runtime that routes its prompt through an ordinary import instead answers the
# program's sentinel and ends the session the program asked to keep.
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
