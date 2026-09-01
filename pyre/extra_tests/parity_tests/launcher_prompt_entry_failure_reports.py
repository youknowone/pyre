# pyre-check: pypy-diverges: pypy3 runs both steps through `run_toplevel`, which
# prints whatever they raise and returns.  So a `SystemExit` from the startup
# file does not end the run there, and a raising hook produces the traceback
# without the fixed line naming it.  The status and the wording are what this
# pins, so the divergence fails the fixture rather than being expressible in it.
#
# CPython-suite gap: `test_cmd_line.test_run_startup` runs a startup file and
# reads its output, nothing anywhere raises from one, and `test_cmd_line` is at
# IMPORTERROR in the baseline besides.  `sys.__interactivehook__` is asserted
# only by `test_site.test_enablerlcompleter`, which checks that `site` registers
# it, never that a prompt calls it or what happens when it fails.
#
# parity-tests reason: both steps report through `pymain_err_print`, which takes
# `SystemExit` before anything else and returns 1 to say the run is over -- so
# `pymain_run_stdin` returns that status instead of opening a prompt.  The hook
# additionally gets a fixed `PySys_WriteStderr` line ahead of the report,
# because a hook is not the program's code and a bare traceback would not say
# where it came from.
import os
import subprocess
import sys
import tempfile

REPORT = 'print("PROMPT REACHED")'


def run(args, env_extra):
    done = subprocess.run(
        [sys.executable, *args],
        input=REPORT + "\n",
        capture_output=True,
        text=True,
        env=dict(os.environ, **env_extra),
    )
    return done.stdout, done.stderr, done.returncode


with tempfile.TemporaryDirectory() as tmp:
    startup = os.path.join(tmp, "exiter.py")
    with open(startup, "w") as f:
        f.write("raise SystemExit(3)\n")
    out, err, code = run(["-i"], {"PYTHONSTARTUP": startup})
    assert "PROMPT REACHED" not in out, (out, err)
    assert code == 3, (code, out, err)

out, err, code = run(
    ["-i", "-c", "import sys\nsys.__interactivehook__ = lambda: 1 / 0\n"], {}
)
assert "Failed calling sys.__interactivehook__" in err, (out, err)
assert "ZeroDivisionError" in err, (out, err)
assert "PROMPT REACHED" in out, (out, err)

print("OK")
