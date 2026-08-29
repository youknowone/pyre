# CPython-suite gap: `test_cmd_line` owns every PYTHONINSPECT test and sits at
# IMPORTERROR in the baseline, and even restored it has one -- a
# `test_inspect_via_env_var` that pipes a program and reads its output.  Nothing
# in the suite reads `sys.flags.interactive` for the variable, and nothing
# checks that the variable alone does *not* open a prompt over a pipe.
#
# parity-tests reason: `-i` sets `inspect` and `interactive`; PYTHONINSPECT sets
# only `inspect`.  `stdin_is_interactive` is `isatty(0) || interactive`, so the
# difference is exactly whether a prompt opens over a stdin that is not a
# terminal.  A launcher that folds the variable into the option reports the
# wrong `sys.flags.interactive` and opens a prompt where every other runtime
# runs the program and exits.
import os
import subprocess
import sys

FLAGS = 'import sys; print("flags", sys.flags.inspect, sys.flags.interactive)'
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


# The variable reaches `inspect` and stops there.
out, err, code = run(["-c", FLAGS], {"PYTHONINSPECT": "1"})
assert "flags 1 0" in out, (out, err)

# `-i` is the spelling that sets both.
out, err, code = run(["-i", "-c", FLAGS], {})
assert "flags 1 1" in out, (out, err)

# Neither is set without either.
out, err, code = run(["-c", FLAGS], {})
assert "flags 0 0" in out, (out, err)

# So the variable alone leaves a piped run with no prompt: `pymain_repl` needs
# `stdin_is_interactive`, and a pipe is not a terminal.
out, err, code = run(["-c", 'print("program ran")'], {"PYTHONINSPECT": "1"})
assert "program ran" in out, (out, err)
assert "PROMPT REACHED" not in out, (out, err)
assert code == 0, (code, err)

# ... and `-i` opens one over the same pipe, because it carries `interactive`.
out, err, code = run(["-i", "-c", 'print("program ran")'], {})
assert "PROMPT REACHED" in out, (out, err)

# The variable does not give the prompt to stdin read as a program either:
# `config_run_code` is false there, so `pymain_repl` returns before it looks.
done = subprocess.run(
    [sys.executable],
    input='print("program ran")\n',
    capture_output=True,
    text=True,
    env=dict(os.environ, PYTHONINSPECT="1"),
)
assert "program ran" in done.stdout, (done.stdout, done.stderr)
assert done.returncode == 0, (done.returncode, done.stderr)

print("OK")
