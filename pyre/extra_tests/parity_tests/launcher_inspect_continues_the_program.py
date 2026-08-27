# CPython-suite gap: exactly one test in the tree pairs `-i` with a program --
# `test_cmd_line.test_run_module_bug1764407`, which runs `-i -m timeit` and
# asserts `__main__.Timer` -- and `test_cmd_line` sits at IMPORTERROR in the
# baseline, so it does not run.  Even restored it covers one half of one shape:
# nothing anywhere pairs `-i` with a program that *fails*, and nothing asserts
# the exit status or when `atexit` runs, so a launcher that starts the prompt in
# a second, freshly-built interpreter passes every test that mentions the
# option.
#
# parity-tests reason: `-i` exists to look at what a program left behind, so
# what it must inherit is the program's own interpreter: `pymain_run_python`
# runs the program, then the prompt, and reaches `Py_FinalizeEx` only after
# both.  That single ordering decides six observable things at once -- whether
# `__main__` survives into the prompt, whether a failure of each kind still
# opens one, what the exit status is, and whether `atexit` runs at the end
# rather than in the middle -- and every one of them is invisible to a runtime
# that never asks.
import subprocess
import sys

REPORT = 'print("PROMPT REACHED")'


def prompt(args, feed=REPORT + "\n"):
    """`(stdout, stderr, returncode)` for a `-i` run driven by piped stdin."""
    # `-i` opens the prompt whether or not stdin is a terminal, so a pipe is
    # enough to drive it and keeps the case readable.
    done = subprocess.run(
        [sys.executable, "-i", *args],
        input=feed,
        capture_output=True,
        text=True,
    )
    return done.stdout, done.stderr, done.returncode


def reached(args, feed=REPORT + "\n"):
    out, err, code = prompt(args, feed)
    return "PROMPT REACHED" in out, code, err


# The program's `__main__` is the prompt's: that is the whole point of the
# option, and a prompt built on a fresh namespace answers NameError instead.
out, err, code = prompt(["-c", "x = 42"], 'print("x =", x)\n')
assert "x = 42" in out, (out, err)
assert code == 0, (code, err)

# A program that failed is what `-i` is most often asked about, so every
# failing shape still opens the prompt and the run ends with the prompt's
# status rather than the program's.
for label, program in (
    ("uncaught exception", "raise ValueError('boom')"),
    ("SystemExit with a code", "raise SystemExit(3)"),
    ("source that will not compile", "bad syntax"),
):
    opened, code, err = reached(["-c", program])
    assert opened, (label, err[-400:])
    assert code == 0, (label, code, err[-400:])

# The failure is still reported -- opening a prompt over it does not swallow it.
_, err, _ = prompt(["-c", "raise ValueError('boom')"])
assert "ValueError: boom" in err, err[-400:]
_, err, _ = prompt(["-c", "raise SystemExit(3)"])
assert "SystemExit: 3" in err, err[-400:]
_, err, _ = prompt(["-c", "bad syntax"])
assert "SyntaxError" in err, err[-400:]

# Startup ran once, so the prompt does not repeat it -- a second pass through
# the import bootstrap reports itself on stderr.
_, err, _ = prompt(["-c", "pass"])
assert "importlib bootstrap failed" not in err, err[-400:]
assert "'import site' failed" not in err, err[-400:]

# `Py_FinalizeEx` runs after the prompt, not between the program and it, so a
# callback the program registered fires when the prompt is done -- and one
# registered at the prompt fires at all.
out, err, code = prompt(
    ["-c", "import atexit; atexit.register(lambda: print('FROM PROGRAM'))"],
    "print('PROMPT REACHED')\n",
)
assert out.index("PROMPT REACHED") < out.index("FROM PROGRAM"), (out, err)
assert code == 0, (code, err)

out, err, code = prompt([], "import atexit\natexit.register(lambda: print('FROM PROMPT'))\n")
assert "FROM PROMPT" in out, (out, err)

# `exit()` at the prompt still delivers its status, and still runs the
# callbacks first.
out, err, code = prompt([], "import atexit\natexit.register(lambda: print('BYE'))\nexit(3)\n")
assert "BYE" in out, (out, err)
assert code == 3, (code, err)
