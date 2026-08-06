"""A child process is launched, talked to over pipes, and waited for.

`subprocess` is two implementations picked apart at import time: the POSIX one
goes through `fork`/`exec` and plain descriptors, the Windows one through
`_winapi` — `CreatePipe` for each redirected stream, `DuplicateHandle` for the
child's inheritable end of it, `CreateProcess`, then `WaitForSingleObject` and
`GetExitCodeProcess`.  What the two owe the caller is the same, and that is
what is asserted here.

The child is this interpreter, so the same program answers on either host.
"""

import collections.abc
import os
import subprocess
import sys
import tempfile

PY = sys.executable
UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}
NON_ASCII = "한글-日本語"


def child(code, *args):
    return [PY, "-c", code, *args]


# An exit code arrives whole, whichever way it is waited for.
assert subprocess.call(child("import sys; sys.exit(3)")) == 3
assert subprocess.check_call(child("pass")) == 0
try:
    subprocess.check_call(child("import sys; sys.exit(1)"))
except subprocess.CalledProcessError as exc:
    assert exc.returncode == 1, exc.returncode
else:
    raise AssertionError("a non-zero exit must raise")

# Each redirected stream is its own pipe, and text mode translates the line
# endings the child wrote whatever they are.
assert subprocess.check_output(child("print('out')"), text=True) == "out\n"
done = subprocess.run(
    child("import sys; print('o'); print('e', file=sys.stderr)"),
    capture_output=True,
    text=True,
)
assert done.stdout == "o\n", repr(done.stdout)
assert done.stderr == "e\n", repr(done.stderr)
assert done.returncode == 0, done.returncode

# `stderr=STDOUT` is the same pipe handed to the child twice, so both streams
# land in `stdout` and nothing is left for `stderr`.
done = subprocess.run(
    child("import sys; sys.stdout.write('o'); sys.stdout.flush(); sys.stderr.write('e')"),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
assert done.stdout == "oe", repr(done.stdout)
assert done.stderr is None, repr(done.stderr)

# What is written to the child's end of the input pipe is what it reads, and
# the pipe closing is the end of file it reads to.
done = subprocess.run(
    child("import sys; sys.stdout.write(sys.stdin.read().upper())"),
    input="abc",
    capture_output=True,
    text=True,
)
assert done.stdout == "ABC", repr(done.stdout)

proc = subprocess.Popen(
    child("import sys; sys.stdout.write(sys.stdin.read()[::-1])"),
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)
out, err = proc.communicate("xyz")
assert (out, err) == ("zyx", None), (out, err)
assert proc.returncode == 0, proc.returncode

# `DEVNULL` is a real handle onto the null device, not a closed stream: the
# child writes to it and lives.
assert subprocess.run(child("print('quiet')"), stdout=subprocess.DEVNULL).returncode == 0

# The environment the child gets is the one it was given and only that.
done = subprocess.run(
    child("import os; print(os.environ.get('PYRE_PROBE'), os.environ.get('PYRE_ABSENT'))"),
    capture_output=True,
    text=True,
    env={**UTF8, "PYRE_PROBE": "value"},
)
assert done.stdout == "value None\n", repr(done.stdout)

# `env` is read as a mapping, which is what `os.environ` is and what the
# documentation asks for — not a `dict`, whose contents it does not have.
os.environ["PYRE_PROBE"] = "inherited"
try:
    done = subprocess.run(
        child("import os; print(os.environ['PYRE_PROBE'])"),
        capture_output=True,
        text=True,
        env=os.environ,
    )
finally:
    del os.environ["PYRE_PROBE"]
assert done.stdout.strip() == "inherited", repr(done.stdout)


class Environ(collections.abc.Mapping):
    """The mapping protocol and nothing else — no dict behind it."""

    contents = {**UTF8, "PYRE_PROBE": "mapped"}

    def __getitem__(self, key):
        return self.contents[key]

    def __iter__(self):
        return iter(self.contents)

    def __len__(self):
        return len(self.contents)


done = subprocess.run(
    child("import os; print(os.environ['PYRE_PROBE'])"),
    capture_output=True,
    text=True,
    env=Environ(),
)
assert done.stdout.strip() == "mapped", repr(done.stdout)

# What is not a mapping is refused rather than launched without an environment.
try:
    subprocess.run(child("pass"), env=object())
except (TypeError, AttributeError):
    pass
else:
    raise AssertionError("a non-mapping env must fail")

# Arguments and environment values are text, and reach the child as the same
# text on a platform whose process arguments are bytes and on one whose are
# wide characters.
done = subprocess.run(
    child("import sys; print(sys.argv[1])", NON_ASCII),
    capture_output=True,
    text=True,
    encoding="utf-8",
    env=UTF8,
)
assert done.stdout.strip() == NON_ASCII, ascii(done.stdout)
done = subprocess.run(
    child("import os; print(os.environ['PYRE_PROBE'])"),
    capture_output=True,
    text=True,
    encoding="utf-8",
    env={**UTF8, "PYRE_PROBE": NON_ASCII},
)
assert done.stdout.strip() == NON_ASCII, ascii(done.stdout)

# `cwd` is the directory the child starts in.
tmp = os.path.realpath(tempfile.gettempdir())
done = subprocess.run(
    child("import os; print(os.path.realpath(os.getcwd()))"),
    capture_output=True,
    text=True,
    cwd=tmp,
)
assert done.stdout.strip() == tmp, (done.stdout, tmp)

# `shell=True` hands the line to the platform's shell rather than naming a
# program, and the shell is the one that splits it.
done = subprocess.run("echo shelled", shell=True, capture_output=True, text=True)
assert done.stdout.strip() == "shelled", repr(done.stdout)

# A name no program answers to fails at the launch, before there is a process
# to have an exit code.
try:
    subprocess.run(["pyre_no_such_program_42"])
except FileNotFoundError as exc:
    assert exc.errno == 2, exc.errno
else:
    raise AssertionError("a missing program must fail")

# A child that outlives its welcome is killed, and the signal or the call that
# killed it is what the exit code then reports.
proc = subprocess.Popen(child("import time; time.sleep(30)"))
assert proc.poll() is None, proc.poll()
proc.terminate()
assert proc.wait() != 0, proc.returncode
assert proc.poll() == proc.returncode

proc = subprocess.Popen(child("import time; time.sleep(30)"))
proc.kill()
assert proc.wait() != 0, proc.returncode

try:
    subprocess.run(child("import time; time.sleep(30)"), timeout=0.5)
except subprocess.TimeoutExpired as exc:
    # The reported timeout is whatever was left of it when the wait gave up,
    # which is the whole of it only where nothing was waited for first.
    assert 0 < exc.timeout <= 0.5, exc.timeout
else:
    raise AssertionError("a child that outlasts the timeout must raise")

# Every launch takes a pair of handles per redirected stream and gives them
# back; a run that leaks them stops launching long before this many.
for _ in range(8):
    assert subprocess.check_output(child("print(1)")).strip() == b"1"

# `os.popen` is the same launch behind a file object, whose `close` reports
# the exit status the way `wait` does — and `None` for the zero that is not
# worth reporting.
QUOTED = '"%s"' % PY  # the interpreter's own directory may hold a space
with os.popen("%s -c \"print('popened')\"" % QUOTED) as pipe:
    assert pipe.read().strip() == "popened"
assert os.popen("%s -c pass" % QUOTED).close() is None
assert os.popen("%s -c \"raise SystemExit(2)\"" % QUOTED, "w").close() != 0

print("OK")
