# pyre-check: gate=1
# `display_exception` sets the sys.last_xxx attributes before it resolves the
# hook, and a hook that fails does not take the exception it was handed with
# it: the failure is named, printed, and the original report follows it under
# `Original exception was:`.  A module run with `-m` ends the same way a script
# does, so the hook it installed is what reports its failure.

import os
import subprocess
import sys
import tempfile


def run(source):
    return subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )


# A hook that raises: both reports reach stderr, in that order.
done = run(
    "import sys\n"
    "def hook(t, v, tb):\n"
    "    raise RuntimeError('hook exploded')\n"
    "sys.excepthook = hook\n"
    "raise ValueError('original boom')\n"
)
assert done.returncode == 1, done.returncode
assert "hook exploded" in done.stderr, done.stderr
assert "Original exception was:" in done.stderr, done.stderr
assert "ValueError: original boom" in done.stderr, done.stderr
assert done.stderr.index("hook exploded") < done.stderr.index(
    "Original exception was:"
), done.stderr

# `display_exception` reads the live `sys.stderr`, so an application that
# replaced the stream reads the whole report on it and the stream underneath
# stays empty.
done = run(
    "import atexit, io, sys\n"
    "cap = io.StringIO()\n"
    "sys.stderr = cap\n"
    "atexit.register(lambda: sys.stdout.write(cap.getvalue()))\n"
    "def hook(t, v, tb):\n"
    "    raise RuntimeError('hook exploded')\n"
    "sys.excepthook = hook\n"
    "raise ValueError('original boom')\n"
)
assert done.returncode == 1, (done.returncode, done.stderr)
assert done.stderr == "", done.stderr
assert "RuntimeError: hook exploded" in done.stdout, done.stdout
assert "Original exception was:" in done.stdout, done.stdout
assert "ValueError: original boom" in done.stdout, done.stdout
assert done.stdout.index("hook exploded") < done.stdout.index(
    "Original exception was:"
), done.stdout


# A hook that is not callable fails the same way.
done = run("import sys\nsys.excepthook = 42\nraise ValueError('original boom')\n")
assert done.returncode == 1, done.returncode
assert "not callable" in done.stderr, done.stderr
assert "ValueError: original boom" in done.stderr, done.stderr

# The four names a post-mortem debugger reads are in place when the hook runs.
done = run(
    "import sys\n"
    "def hook(t, v, tb):\n"
    "    print(sys.last_type is t, sys.last_value is v, sys.last_traceback is tb)\n"
    "    print(sys.last_exc is v)\n"
    "sys.excepthook = hook\n"
    "raise ValueError('boom')\n"
)
assert done.returncode == 1, (done.returncode, done.stderr)
assert done.stdout.split() == ["True", "True", "True", "True"], done.stdout

# `-m` reports its failure through the same route, so a hook the module
# installed is what runs.
with tempfile.TemporaryDirectory() as directory:
    package = os.path.join(directory, "pyre_hook_pkg")
    os.mkdir(package)
    open(os.path.join(package, "__init__.py"), "w").close()
    with open(os.path.join(package, "__main__.py"), "w") as handle:
        handle.write(
            "import sys\n"
            "def hook(t, v, tb):\n"
            "    print('SENTINEL', t.__name__, v)\n"
            "sys.excepthook = hook\n"
            "raise ValueError('m-mode boom')\n"
        )
    done = subprocess.run(
        [sys.executable, "-m", "pyre_hook_pkg"],
        capture_output=True,
        text=True,
        cwd=directory,
    )
assert done.returncode == 1, (done.returncode, done.stderr)
assert done.stdout.strip() == "SENTINEL ValueError m-mode boom", (
    done.stdout,
    done.stderr,
)

print("OK")
