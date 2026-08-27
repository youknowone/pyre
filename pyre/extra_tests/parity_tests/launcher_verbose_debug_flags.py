# pyre-check: pypy-diverges: pypy3 counts `-d` the way it counts `-v`, so it
# answers 2 for `-dd` where 3.14 saturates `sys.flags.debug` at 1, and its
# `parse_env` folds on `if v:`, so a `=0` from either variable enables the flag
# there and leaves it alone here.  Its verbose banner diverges twice more:
# `app_main.py:1032` reaches `print_banner` under a bare `if verbose:`, with no
# `quiet` test, so `-q -v` still prints both lines there; and `print_banner`
# writes through `sys.stderr`, so a `sitecustomize` that rebinds the attribute
# captures the banner that `fprintf(stderr, ...)` sends to the descriptor.
#
# CPython-suite gap: `test_site` and `test_cmd_line` reach `-v` only by running
# a subprocess with it, and both modules are outside the gated set, so a
# launcher that refuses the option outright fails them for a reason that never
# names the option.  Nothing in the gated set passes `-v` or `-d` at all.
#
# parity-tests reason: `-v` and `-d` are the two repeatable options whose whole
# observable is a `sys.flags` field, and the two fields disagree about what
# repeating means.  `initconfig.c PYCONFIG_SPEC` declares `verbose` UINT, so
# `-vv` reports 2; it declares `parser_debug` BOOL, so `-dd` reports 1 even
# though the config counted to 2.  A port that treats the pair alike is wrong
# about one of them whichever way it picks -- pypy3 counts both and so answers
# 2 for `-dd`.
#
# `-v` also has a second half that is not a flag: `importlib._bootstrap`'s
# `_verbose_message` compares `sys.flags.verbose` against the verbosity a
# message asks for, so setting the field is what makes the import trace appear.
# The trace's content is the import machinery's, not the launcher's, so what is
# pinned here is that it reaches stderr at all and that stdout stays clean.
import os
import subprocess
import sys
import tempfile

REPORT = "import sys; print('%d %d' % (sys.flags.verbose, sys.flags.debug))"

# Both variables this file pins are read from the environment, so a parent that
# already sets one answers every case with its own value instead of the case's
# -- `flags()` alone reports it, and a set PYTHONVERBOSE also puts a trace on the
# stderr the quiet rows require to be empty.  Removed for every child; a case
# that names one puts it back.
SCRUBBED = {name: None for name in ("PYTHONVERBOSE", "PYTHONDEBUG")}


def child_env(env=None):
    """The parent environment with the two variables this file owns settled."""
    environ = dict(os.environ)
    for name, value in {**SCRUBBED, **(env or {})}.items():
        if value is None:
            environ.pop(name, None)
        else:
            environ[name] = value
    return environ


def flags(args=(), env=None):
    """`(verbose, debug)` as a child of this interpreter reports them."""
    out = subprocess.run(
        [sys.executable, *args, "-c", REPORT],
        capture_output=True,
        text=True,
        env=child_env(env),
    )
    assert out.returncode == 0, (args, env, out.returncode, out.stderr[-400:])
    # `-v` writes its trace to stderr, so the report is still the last line of
    # stdout on its own.
    return tuple(int(part) for part in out.stdout.strip().splitlines()[-1].split())


assert flags() == (0, 0)

# Repeating counts, and the two fields part company at the second one.
assert flags(["-v"]) == (1, 0)
assert flags(["-vv"]) == (2, 0)
assert flags(["-vvv"]) == (3, 0)
assert flags(["-d"]) == (0, 1)
assert flags(["-dd"]) == (0, 1)
assert flags(["-v", "-d"]) == (1, 1)

# The environment variables fold with `max`, so a variable can raise a count
# the command line already set but never lower it.
assert flags(env={"PYTHONVERBOSE": "1"}) == (1, 0)
assert flags(env={"PYTHONVERBOSE": "3"}) == (3, 0)
assert flags(["-v"], env={"PYTHONVERBOSE": "3"}) == (3, 0)
assert flags(["-vvvv"], env={"PYTHONVERBOSE": "2"}) == (4, 0)
# A value that is not a non-negative integer counts as 1, the way every other
# `_Py_get_env_flag` variable reads one.
assert flags(env={"PYTHONVERBOSE": "junk"}) == (1, 0)
assert flags(env={"PYTHONVERBOSE": "-2"}) == (1, 0)
# An empty value is unset, not zero.
assert flags(env={"PYTHONVERBOSE": ""}) == (0, 0)
# `=0` is a value the fold reads and leaves alone.
assert flags(env={"PYTHONVERBOSE": "0"}) == (0, 0)
# `_Py_str_to_int` rejects anything outside the C `int` range as overflow
# (`preconfig.c:554`), and an unreadable value is what becomes 1, so INT_MAX is
# the last count the variable can spell and everything past it reads as junk.
# The ceiling is the C type's, not the width of whatever field holds the count.
assert flags(env={"PYTHONVERBOSE": "2147483647"}) == (2147483647, 0)
assert flags(env={"PYTHONVERBOSE": "2147483648"}) == (1, 0)
assert flags(env={"PYTHONVERBOSE": "4294967295"}) == (1, 0)

# `debug` saturates on the way out, so the variable cannot push it past 1.
assert flags(env={"PYTHONDEBUG": "2"}) == (0, 1)
assert flags(env={"PYTHONDEBUG": "junk"}) == (0, 1)
assert flags(env={"PYTHONDEBUG": ""}) == (0, 0)
# The saturation is the only thing `debug` does differently: `=0` is still a
# value `_Py_get_env_flag` reads and leaves alone, so naming the variable is not
# by itself a way to ask for the flag.  `test_cmd_line.test_sys_flags_set`
# compares against `int(bool(value))` but never passes it a `"0"`, so the case
# is decided by the fold rather than by that test.
assert flags(env={"PYTHONDEBUG": "0"}) == (0, 0)
assert flags(env={"PYTHONDEBUG": "-1"}) == (0, 1)

# `-E` drops both variables; `-I` implies it.
assert flags(["-E"], env={"PYTHONVERBOSE": "3", "PYTHONDEBUG": "1"}) == (0, 0)
assert flags(["-I"], env={"PYTHONVERBOSE": "3", "PYTHONDEBUG": "1"}) == (0, 0)
# The command line survives `-E` -- only the variable half is dropped.
assert flags(["-E", "-vv"], env={"PYTHONVERBOSE": "3"}) == (2, 0)


def import_trace(args):
    """stderr and stdout of a child that imports one stdlib module."""
    out = subprocess.run(
        [sys.executable, *args, "-c", "import json; print('done')"],
        capture_output=True,
        text=True,
        env=child_env(),
    )
    assert out.returncode == 0, (args, out.returncode, out.stderr[-400:])
    return out.stderr, out.stdout


quiet_err, quiet_out = import_trace([])
assert quiet_err == "", quiet_err
assert quiet_out == "done\n", quiet_out

loud_err, loud_out = import_trace(["-v"])
# The program's own output is untouched: the trace goes to stderr alone.
assert loud_out == "done\n", loud_out
# `_verbose_message` writes `import <name> # <origin>` lines, and the module
# the child asked for is one of them.
assert "import 'json'" in loud_err, loud_err[-600:]
assert loud_err.count("\n") > 20, loud_err.count("\n")


# `_Py_str_to_int` runs `strtol`, which steps over leading whitespace before the
# digits and then refuses whatever it stopped short of, so a value padded on the
# LEFT is the count it spells.  A parse that rejects both pads reports 1 -- the
# same answer as `PYTHONVERBOSE=abc` -- and silently downgrades verbosity 2.
# The right-hand pad is deliberately not pinned: `strtol` stops at it and
# CPython answers 1 where pypy3's `int()` answers 2.
assert flags(env={"PYTHONVERBOSE": " 2"}) == (2, 0)
assert flags(env={"PYTHONVERBOSE": "+2"}) == (2, 0)
assert flags(env={"PYTHONVERBOSE": "0x2"}) == (1, 0)


def stdin_run(args):
    """stderr and stdout of a child whose program arrives on a pipe."""
    out = subprocess.run(
        [sys.executable, *args],
        input="print('done')\n",
        capture_output=True,
        text=True,
        env=child_env(),
    )
    assert out.returncode == 0, (args, out.returncode, out.stderr[-400:])
    return out.stderr, out.stdout


# `run_command_line` heads a non-interactive stdin run with the banner when
# `verbose` is set, so the trace that follows says which interpreter produced
# it.  Only the presence is pinned: the three implementations spell their own
# identification differently, and the line is not first -- the trace of the
# imports that precede it already is.
COPYRIGHT_LINE = (
    'Type "help", "copyright", "credits" or "license" for more information.'
)

banner_err, banner_out = stdin_run(["-v"])
assert banner_out == "done\n", banner_out
assert any(
    line.startswith(("Python ", "pyre ")) for line in banner_err.splitlines()
), banner_err[:400]
# `print_banner(not no_site)` -- the second line is spelled the same everywhere,
# because what it names is the four builtins `site` installs rather than
# anything about the interpreter.
banner_lines = banner_err.splitlines()
assert COPYRIGHT_LINE in banner_lines, banner_err[:400]
# The banner follows the startup import trace rather than heading it: what it
# labels is that trace, so it is printed once `site` and the warnings bootstrap
# have run, not on the way in.  Pinned as "something precedes it" rather than an
# index -- the three implementations import a different number of modules -- and
# the notice follows the identification rather than merely appearing somewhere.
identification = next(
    i for i, line in enumerate(banner_lines) if line.startswith(("Python ", "pyre "))
)
assert identification > 0, banner_err[:400]
assert banner_lines.index(COPYRIGHT_LINE) > identification, banner_err[:400]
# `-S` installs none of them, so the line goes with them while the
# identification stays.
site_err, site_out = stdin_run(["-S", "-v"])
assert site_out == "done\n", site_out
assert any(
    line.startswith(("Python ", "pyre ")) for line in site_err.splitlines()
), site_err[:400]
assert COPYRIGHT_LINE not in site_err.splitlines(), site_err[:400]
# `pymain_header` tests `quiet` before it tests anything else, so `-q` drops the
# whole banner in every mode that would otherwise print one -- this arm included,
# where `-q` otherwise has nothing to suppress.  pypy3 has no such test on its
# stdin branch and prints both lines anyway.
quiet_banner_err, quiet_banner_out = stdin_run(["-q", "-v"])
assert quiet_banner_out == "done\n", quiet_banner_out
assert not any(
    line.startswith(("Python ", "pyre ")) for line in quiet_banner_err.splitlines()
), quiet_banner_err[:400]
assert COPYRIGHT_LINE not in quiet_banner_err.splitlines(), quiet_banner_err[:400]


def banner_destination():
    """`(reached fd 2, reached a replaced sys.stderr)` for the verbose banner."""
    with tempfile.TemporaryDirectory() as root:
        captured = os.path.join(root, "captured")
        with open(os.path.join(root, "sitecustomize.py"), "w") as handle:
            handle.write(
                "import sys\n"
                "sys.stderr = open(%r, 'w', buffering=1)\n" % captured
            )
        out = subprocess.run(
            [sys.executable, "-v"],
            input="print('done')\n",
            capture_output=True,
            text=True,
            env=child_env({"PYTHONPATH": root}),
        )
        assert out.returncode == 0, (out.returncode, out.stderr[-400:])
        with open(captured, encoding="utf-8", errors="replace") as handle:
            replaced = handle.read()

    def has_banner(text):
        return any(
            line.startswith(("Python ", "pyre ")) for line in text.splitlines()
        )

    return has_banner(out.stderr), has_banner(replaced)


# `fprintf(stderr, ...)`: the banner goes to the process descriptor, so a
# `sitecustomize` that rebinds `sys.stderr` during `import site` -- which runs
# before the banner -- does not redirect it.  pypy3 writes it through the live
# `sys.stderr` and so follows the rebind.
assert banner_destination() == (True, False), banner_destination()
# Without `-v` there is no banner, and a pipe still is not a prompt.
plain_err, plain_out = stdin_run([])
assert plain_out == "done\n", plain_out
assert not any(
    line.startswith(("Python ", "pyre ")) for line in plain_err.splitlines()
), plain_err[:400]

print("OK")
