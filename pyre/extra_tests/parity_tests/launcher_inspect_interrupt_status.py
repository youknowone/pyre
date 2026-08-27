# pyre-check: platforms=linux,darwin
# pyre-check: pypy-diverges: `app_main.py` treats the interactive prompt as the
# end of the story -- it returns the prompt's own status and keeps no record of
# an interrupt the program reported -- so pypy3 answers 0 for every `-i` row
# below where 3.14 dies of SIGINT.  Its no-`-i` control agrees at -2, which is
# what places the divergence in the prompt rather than in the interrupt.
#
# CPython-suite gap: the suite pins the interrupt status only for a program run
# without `-i` -- `test_cmd_line.test_keyboard_interrupt_exit_code` and the
# `test_subprocess` SIGINT cases -- and pairs `-i` with a program exactly once
# anywhere in the tree (`test_run_module_bug1764407`, which asserts a name, not
# a status).  So nothing covers what the prompt does with an interrupt raised
# before it opened.
#
# parity-tests reason: `_PyErr_PrintEx` records an unhandled `KeyboardInterrupt`
# and `Py_RunMain` consults that record after the prompt and after
# `Py_FinalizeEx`, so the interrupt outlives a prompt opened over it.  That is
# what keeps `-i` honest for the shell: a run that was interrupted still reports
# itself as interrupted, and a `python -i` in a script or CI job is not silently
# turned into a success by the prompt that followed.  An explicit `exit(n)`
# answers the interrupt instead, `exit(0)` included, because `SystemExit` leaves
# through `Py_Exit` without reaching the check.
#
# `subprocess` reports a signal death as the negative signal number, so SIGINT
# is -2 here rather than the 128+2 a shell prints.  Windows encodes it
# differently again, hence the `platforms` header.
import signal
import subprocess
import sys

INTERRUPTED = -signal.SIGINT


def run(args, feed=""):
    return subprocess.run(
        [sys.executable, *args], input=feed, capture_output=True, text=True
    ).returncode


# The control: without `-i` every runtime already dies of the interrupt, so the
# rows below are about the prompt and nothing else.
assert run(["-c", "raise KeyboardInterrupt"]) == INTERRUPTED

# A prompt that simply ended does not answer the interrupt, so the run still
# reports itself as interrupted.
assert run(["-i", "-c", "raise KeyboardInterrupt"]) == INTERRUPTED

# An interrupt raised at the prompt itself is recorded the same way.
assert run(["-i", "-c", "pass"], "raise KeyboardInterrupt\n") == INTERRUPTED

# An explicit status answers it -- including a successful one, which is the
# case that cannot be explained by "the largest status wins".
assert run(["-i", "-c", "raise KeyboardInterrupt"], "exit(0)\n") == 0
assert run(["-i", "-c", "raise KeyboardInterrupt"], "exit(3)\n") == 3

# A different exception is not an interrupt: the prompt's own clean end stands.
assert run(["-i", "-c", "raise ValueError('boom')"]) == 0

print("OK")
