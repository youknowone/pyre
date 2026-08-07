"""An uncaught KeyboardInterrupt kills the process by SIGINT, not with status 1.

`app_main.py:1133-1153` restores `SIG_DFL` and then signals the process itself —
`_signal.raise_signal(SIGINT)` on win32, `os.kill(os.getpid(), SIGINT)` elsewhere
— so a shell sees the interrupt rather than an ordinary error exit. The child is
spawned because the fixture itself has to exit 0.

`-c` and `-m` reach that ending through separate call sites, so both are
asserted; the remaining cases pin the exit paths this must not have moved.
"""

import os
import pathlib
import signal
import subprocess
import sys
import tempfile


def run_child(source):
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


child = run_child("raise KeyboardInterrupt")
if sys.platform == "win32":
    # STATUS_CONTROL_C_EXIT: the CRT's default SIGINT action, reached through
    # `raise()` rather than `os.kill`, which would terminate with 2 instead.
    assert child.returncode == 0xC000013A, child.returncode
else:
    assert child.returncode == -signal.SIGINT, child.returncode
# The traceback is printed before the process dies, on this path too.
assert "Traceback" in child.stderr, child.stderr
assert "KeyboardInterrupt" in child.stderr, child.stderr

# An ordinary uncaught exception still exits 1 and still prints.
child = run_child("raise RuntimeError('boom')")
assert child.returncode == 1, child.returncode
assert "RuntimeError: boom" in child.stderr, child.stderr

# SystemExit still carries its own code.
child = run_child("raise SystemExit(3)")
assert child.returncode == 3, child.returncode

# A KeyboardInterrupt the script catches must not kill anything.
child = run_child(
    "try:\n"
    "    raise KeyboardInterrupt\n"
    "except KeyboardInterrupt:\n"
    "    print('caught')\n"
)
assert child.returncode == 0, child.returncode
assert child.stdout.strip() == "caught", child.stdout

# `-m` runs the module through a separate call site, so it needs the same two
# statuses of its own: a module that raises the interrupt, and one that fails to
# import at all.
with tempfile.TemporaryDirectory() as module_dir:
    pathlib.Path(module_dir, "kbd_interrupt_module.py").write_text(
        "raise KeyboardInterrupt\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = module_dir + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run(
        [sys.executable, "-m", "kbd_interrupt_module"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
if sys.platform == "win32":
    assert child.returncode == 0xC000013A, child.returncode
else:
    assert child.returncode == -signal.SIGINT, child.returncode
assert "Traceback" in child.stderr, child.stderr
assert "KeyboardInterrupt" in child.stderr, child.stderr

child = subprocess.run(
    [sys.executable, "-m", "this_module_does_not_exist_zzz"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)
assert child.returncode == 1, child.returncode

print("OK")
