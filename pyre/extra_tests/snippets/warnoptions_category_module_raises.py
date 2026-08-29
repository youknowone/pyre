# pyre-check: gate=1
# pyre-check: pypy-diverges: pins the program still running after a -W category module raises; pypy3 catches ImportError alone there, so the RuntimeError ends the process
# CPython-suite gap: `test_warnings` drives `_setoption` in-process, where a
# category module that raises reaches the caller as `_OptionError` or is not
# reached at all.  Nothing in the suite starts an interpreter whose `-W` names
# a module that raises at import, so the startup arm around it is untested.
#
# parity-tests reason: the arm is startup code, not warnings code -- it decides
# whether a program runs at all.  Dropping the failure runs the program with
# the filters silently half-applied; carrying it out of the bootstrap ends a
# program the reference runs.  Both are invisible to an in-process test.

"""A `-W` category module that raises is reported, and the program still runs."""

import os
import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(tmp, "raising_category.py"), "w") as module:
        module.write('raise RuntimeError("raised-from-the-category-module")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = tmp + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-W", "ignore::raising_category.W", "-c", "print('RAN')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

# The filter never takes effect, but the program it was meant to apply to is
# not what the failure is about, so it runs and reports success.
assert result.returncode == 0, (result.returncode, result.stderr)
assert result.stdout.strip() == b"RAN", result.stdout
# And the failure is not silent: whatever the module raised names itself on
# stderr, so a filter that did not apply is not mistaken for one that did.
assert b"raised-from-the-category-module" in result.stderr, result.stderr

print("OK")
