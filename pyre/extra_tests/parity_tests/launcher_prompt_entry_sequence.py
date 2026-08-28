# CPython-suite gap: `test_cmd_line` is the only module that reaches these, and
# it sits at IMPORTERROR in the baseline, so none of it runs.  Even restored,
# `test_run_startup` there checks that a startup file is executed and nothing
# checks which prompts run it; `sys.__interactivehook__` is asserted only by
# `test_site.test_enablerlcompleter` -- that `site` *registers* it, never that a
# prompt calls it.
#
# parity-tests reason: the two steps are what a prompt does before it reads its
# first line, and both are invisible to a runtime that opens one without them.
# `pymain_run_stdin` runs the startup file and then the hook; `pymain_repl` --
# the prompt `-i` opens over a program -- runs only the hook, which is why a
# `-i` script run must not execute a startup file a second time.
import os
import subprocess
import sys
import tempfile

REPORT = 'print("PROMPT REACHED")'


def run(args, env_extra, feed=REPORT + "\n"):
    env = dict(os.environ, **env_extra)
    done = subprocess.run(
        [sys.executable, *args],
        input=feed,
        capture_output=True,
        text=True,
        env=env,
    )
    return done.stdout, done.stderr, done.returncode


with tempfile.TemporaryDirectory() as tmp:
    startup = os.path.join(tmp, "startup.py")
    with open(startup, "w") as f:
        f.write('print("STARTUP", __file__)\n')

    # The prompt that reads stdin runs it.  `-i` makes a pipe drive that prompt
    # without a terminal, which is the same branch a tty takes.
    out, err, code = run(["-i"], {"PYTHONSTARTUP": startup})
    assert "STARTUP" in out, (out, err)
    # `_PyRun_SimpleFileObject` binds `__file__` for the file it runs, because
    # the prompt's `__main__` has none of its own.
    assert startup in out, (out, err)
    assert "PROMPT REACHED" in out, (out, err)
    assert code == 0, (code, err)

    # ... and unbinds it again, so the prompt that follows has no `__file__`.
    out, err, code = run(
        ["-i"],
        {"PYTHONSTARTUP": startup},
        'print("FILE", "__file__" in dir())\n',
    )
    assert "FILE False" in out, (out, err)

    # The prompt that follows a program is `pymain_repl`, which does not.
    for args in (["-i", "-c", "pass"], ["-i", os.path.join(tmp, "prog.py")]):
        with open(os.path.join(tmp, "prog.py"), "w") as f:
            f.write('print("program ran")\n')
        out, err, code = run(args, {"PYTHONSTARTUP": startup})
        assert "STARTUP" not in out, (args, out, err)
        assert "PROMPT REACHED" in out, (args, out, err)

    # `-E` refuses the variable the way it refuses every other one.
    out, err, code = run(["-E", "-i"], {"PYTHONSTARTUP": startup})
    assert "STARTUP" not in out, (out, err)
    assert "PROMPT REACHED" in out, (out, err)

    # A file that will not open is reported by name and the prompt still opens.
    missing = os.path.join(tmp, "not-there.py")
    out, err, code = run(["-i"], {"PYTHONSTARTUP": missing})
    assert "Could not open PYTHONSTARTUP" in err, (out, err)
    assert missing in err, (out, err)
    assert "PROMPT REACHED" in out, (out, err)
    assert code == 0, (code, err)

    # A startup file that raises is reported, and the prompt opens over it.
    raiser = os.path.join(tmp, "raiser.py")
    with open(raiser, "w") as f:
        f.write('raise ValueError("from startup")\n')
    out, err, code = run(["-i"], {"PYTHONSTARTUP": raiser})
    assert "ValueError: from startup" in err, (out, err)
    assert "PROMPT REACHED" in out, (out, err)

# Every prompt calls the hook, including the one that opens over a program.
HOOK = """
import sys
def hook():
    print("HOOK CALLED")
sys.__interactivehook__ = hook
"""
out, err, code = run(["-i", "-c", HOOK], {})
assert "HOOK CALLED" in out, (out, err)
assert "PROMPT REACHED" in out, (out, err)
assert out.index("HOOK CALLED") < out.index("PROMPT REACHED"), out

# A hook that raises is reported, and the prompt opens anyway.
out, err, code = run(
    ["-i", "-c", "import sys\nsys.__interactivehook__ = lambda: 1 / 0\n"],
    {},
)
assert "ZeroDivisionError" in err, (out, err)
assert "PROMPT REACHED" in out, (out, err)

# A `sys` with no hook at all is the ordinary `-S` shape, not a failure.
out, err, code = run(["-S", "-i", "-c", "pass"], {})
assert "PROMPT REACHED" in out, (out, err)
assert code == 0, (code, err)

print("OK")
