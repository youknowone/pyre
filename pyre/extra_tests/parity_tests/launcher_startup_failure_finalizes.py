# CPython-suite gap: `test_cmd_line` and `test_cmd_line_script` are the two
# modules that would own this and both sit at IMPORTERROR in the baseline, so
# neither runs.  Even restored, what they assert about these shapes is the
# message and the exit status -- `test_cmd_line_script.test_dash_c_error` and
# friends -- and nothing anywhere runs a `sitecustomize` alongside them, so a
# runtime that skips finalization on the way out of a startup failure passes
# every one of them.
#
# parity-tests reason: `Py_RunMain` reports whatever `pymain_run_python` failed
# at and then calls `Py_FinalizeEx` regardless, so a failure that happens after
# initialisation still runs the `atexit` callbacks, tears the modules down and
# flushes the streams.  `site` is part of that initialisation, which is what
# makes the rule observable: `sitecustomize` has already had its chance to
# register a callback by the time the program is even read, so "the program
# never ran" is not a reason to skip the callbacks.  A runtime that exits
# directly from these paths loses them silently -- nothing is printed, and the
# exit status is the one the caller expected anyway.
import os
import subprocess
import sys
import tempfile

MARK = "ATEXIT RAN"


def _fixture_dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "sitecustomize.py"), "w") as fp:
        fp.write("import atexit, sys\n")
        fp.write(f"atexit.register(lambda: sys.stderr.write({MARK!r} + chr(10)))\n")
    with open(os.path.join(d, "badsyn.py"), "w") as fp:
        fp.write("bad syntax\n")
    return d


DIR = _fixture_dir()


def run(args, feed=None):
    """`(finalized, returncode)` for a child whose `site` registers a callback."""
    env = dict(os.environ, PYTHONPATH=DIR)
    # `-S` would take the whole mechanism away, so make sure nothing in the
    # ambient environment has switched site off for this child.
    env.pop("PYTHONNOUSERSITE", None)
    done = subprocess.run(
        [sys.executable, *args],
        input=feed,
        capture_output=True,
        text=True,
        cwd=DIR,
        env=env,
    )
    return MARK in done.stderr, done.returncode


# The control: a program that ran and raised is finalized by every runtime, so
# a failure below is about *which* paths finalize, not about whether the
# callback works at all.
finalized, code = run(["-c", "raise ValueError('boom')"])
assert finalized, "sitecustomize callback never ran even on the control path"
assert code == 1, code

# Source that will not compile: reached after initialisation on every one of
# the three entry points that take source.
for label, args, feed in (
    ("-c", ["-c", "bad syntax"], None),
    ("script", ["badsyn.py"], None),
    ("stdin", [], "bad syntax\n"),
):
    finalized, code = run(args, feed)
    assert finalized, f"{label}: startup failure skipped finalization"
    assert code == 1, (label, code)

# A script that cannot be opened is reported by a live interpreter too:
# `pymain_run_file` opens it only once initialisation is complete.  Its status
# is `EXIT_STATUS_ERROR`, distinct from the 1 a program that ran and failed
# exits with.
finalized, code = run(["no_such_file_here.py"])
assert finalized, "unopenable script skipped finalization"
assert code == 2, code

print("OK")
