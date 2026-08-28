# CPython-suite gap: `test_faulthandler` and `test_cmd_line` both reach these
# through subprocesses, and neither module is in the gated set, so a launcher
# that silently drops `-X faulthandler` or `-X pycache_prefix` fails nothing.
# The one gated module that touches the second, `test_importlib`, only reads
# `sys.pycache_prefix` back -- it never asks a launcher to set it.
#
# parity-tests reason: these are the two `-X` options pypy3 implements that
# pyre did not, and each carries a rule the obvious implementation gets wrong.
#
# PYTHONFAULTHANDLER takes the *presence* fold, not the integer one: `=0`
# enables it, the same as any other non-empty value, and only an empty value
# reads as unset.  A port that folds it the way PYTHONOPTIMIZE folds is wrong
# about exactly one spelling, and it is the spelling someone writes when they
# mean to turn it off.
#
# `-X pycache_prefix` is three-state, and the third state has no value to
# carry: naming the option at all settles the question, so `-X pycache_prefix`
# and `-X pycache_prefix=` both leave the prefix unset *and* keep
# PYTHONPYCACHEPREFIX from supplying one.  Only an option nobody named reaches
# the variable.  A two-state port -- an `Option<String>` filled from the `=`
# tail -- collapses the bare spelling into "absent" and lets the environment
# through, which is the one case where a command line asking for no prefix
# gets one.
#
# The prefix is also not merely reported: `_bootstrap_external`
# `cache_from_source` computes the bytecode path from it, so setting the field
# is what redirects the write.  That is the half a packager depends on, and
# `pip` byte-compiles what it installs, so the last check imports a fresh
# module and looks at where its `.pyc` actually landed.
import ast
import os
import subprocess
import sys
import tempfile

# The names under test are read from the environment, so a harness that
# already carries one would decide the answer.  Every child starts from an
# environment with all four removed and gets back only what a case asks for.
SCRUBBED = {
    name: None
    for name in (
        "PYTHONFAULTHANDLER",
        "PYTHONPYCACHEPREFIX",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONDEVMODE",
    )
}


def child(args=(), env=None, code=None):
    """The last stdout line of this interpreter run with `args` and `env`."""
    environ = dict(os.environ)
    for name, value in {**SCRUBBED, **(env or {})}.items():
        if value is None:
            environ.pop(name, None)
        else:
            environ[name] = value
    out = subprocess.run(
        [sys.executable, *args, "-c", code or REPORT],
        capture_output=True,
        text=True,
        env=environ,
    )
    assert out.returncode == 0, (args, env, out.returncode, out.stderr[-400:])
    # Only the last line: `-X dev` installs a warning filter, and a child that
    # prints nothing at all is a legitimate case here.
    lines = out.stdout.splitlines()
    return lines[-1].strip() if lines else ""


REPORT = (
    "import faulthandler, sys;"
    "print(faulthandler.is_enabled(), repr(sys.pycache_prefix))"
)


def state(args=(), env=None):
    enabled, prefix = child(args, env).split(" ", 1)
    return enabled == "True", ast.literal_eval(prefix)


assert state() == (False, None)

# -- faulthandler ---------------------------------------------------------
# `run_command_line` installs nothing at all unless `'faulthandler' in
# sys.builtin_module_names`, and that guard is not decoration: pypy drops the
# module from `working_modules` on Windows (`pypy/config/pypyoption.py:78`), so
# on that one pairing every spelling below is a no-op by upstream's own rule
# rather than by a divergence.  Taking the guard here keeps the lane asserting
# something on that build instead of skipping it, and costs nothing where the
# module is present -- 3.14, pypy3 off Windows, and pyre on every platform all
# report it builtin, so `INSTALLS` is True for each of them.
INSTALLS = "faulthandler" in sys.builtin_module_names

assert state(["-X", "faulthandler"]) == (INSTALLS, None)
# `run_command_line` tests `'faulthandler' in sys._xoptions`, so the option
# answers to its KEY: every value installs the handlers, `=0` included, which is
# not a way to ask for them to stay off.  An exact match on the whole option
# passes the bare spelling above and silently drops these three.
for spelled in ("faulthandler=1", "faulthandler=0", "faulthandler="):
    assert state(["-X", spelled]) == (INSTALLS, None), spelled
# Keyed, not merely prefixed: a longer name is a different option.
assert state(["-X", "faulthandlerx"]) == (False, None)
# The rule is per-option, not a blanket: `-X dev` is read by VALUE, so pypy3
# leaves developer mode off for `-X dev=1` while CPython turns it on.  pyre
# follows pypy3 there, so that contrast cannot be asserted here -- the two
# oracles agree on every line this file does pin.
# Developer mode asks for it too, so it is not only the option that installs --
# and it asks in either of its spellings, which pins that the variable fold for
# `dev_mode` happens before this one reads it.
assert state(["-X", "dev"]) == (INSTALLS, None)
assert state(env={"PYTHONDEVMODE": "1"}) == (INSTALLS, None)
# `-X dev` survives `-E`, so an environment that names neither still installs.
assert state(["-E", "-X", "dev"]) == (INSTALLS, None)
assert state(env={"PYTHONFAULTHANDLER": "1"}) == (INSTALLS, None)
# The presence fold, which is what separates this variable from PYTHONOPTIMIZE
# and PYTHONDONTWRITEBYTECODE: a zero still enables, an empty value does not.
assert state(env={"PYTHONFAULTHANDLER": "0"}) == (INSTALLS, None)
assert state(env={"PYTHONFAULTHANDLER": "anything"}) == (INSTALLS, None)
assert state(env={"PYTHONFAULTHANDLER": ""}) == (False, None)
# `-E` and `-I` drop the variable; the option survives both.
assert state(["-E"], env={"PYTHONFAULTHANDLER": "1"}) == (False, None)
assert state(["-I"], env={"PYTHONFAULTHANDLER": "1"}) == (False, None)
assert state(["-E", "-X", "faulthandler"], env={"PYTHONFAULTHANDLER": ""}) == (
    INSTALLS,
    None,
)

# -- pycache_prefix -------------------------------------------------------
assert state(["-X", "pycache_prefix=/tmp/from-option"]) == (False, "/tmp/from-option")
assert state(env={"PYTHONPYCACHEPREFIX": "/tmp/from-env"}) == (False, "/tmp/from-env")
# Stored the way it was written: a relative path is not resolved against the
# working directory on the way in.
assert state(["-X", "pycache_prefix=relative/dir"]) == (False, "relative/dir")
# The option outranks the variable, and `-E` drops the variable outright.
assert state(
    ["-X", "pycache_prefix=/tmp/from-option"],
    env={"PYTHONPYCACHEPREFIX": "/tmp/from-env"},
) == (False, "/tmp/from-option")
assert state(["-E"], env={"PYTHONPYCACHEPREFIX": "/tmp/from-env"}) == (False, None)
# The three states.  Both spellings that name the option without naming a
# directory leave the prefix unset *and* suppress the variable; an empty
# variable is simply unset.
assert state(["-X", "pycache_prefix"]) == (False, None)
assert state(["-X", "pycache_prefix="]) == (False, None)
assert state(["-X", "pycache_prefix"], env={"PYTHONPYCACHEPREFIX": "/tmp/from-env"}) == (
    False,
    None,
)
assert state(["-X", "pycache_prefix="], env={"PYTHONPYCACHEPREFIX": "/tmp/from-env"}) == (
    False,
    None,
)
assert state(env={"PYTHONPYCACHEPREFIX": ""}) == (False, None)


def where_the_pyc_lands(use_prefix):
    """`(under the prefix, beside the source)` after importing a fresh module."""
    with tempfile.TemporaryDirectory() as root:
        source_dir = os.path.join(root, "src")
        prefix = os.path.join(root, "cache")
        os.mkdir(source_dir)
        with open(os.path.join(source_dir, "pcpmod.py"), "w") as handle:
            handle.write("VALUE = 1\n")
        # A fresh name in a fresh directory, so the import has to compile and
        # has nothing cached to read instead.
        child(
            ["-X", "pycache_prefix=" + prefix] if use_prefix else [],
            code="import sys; sys.path.insert(0, %r); import pcpmod" % source_dir,
        )

        # Only this module's bytecode is counted.  A prefix collects the whole
        # startup import as well -- `encodings`, `linecache` -- and how much of
        # that a runtime recompiles is its own business, so a total would be
        # counting the stdlib rather than the redirect.
        def pcpmod_pyc_count(top):
            return sum(
                name.startswith("pcpmod") and name.endswith(".pyc")
                for _, _, names in os.walk(top)
                for name in names
            )

        under = pcpmod_pyc_count(prefix) if os.path.isdir(prefix) else 0
        return under, pcpmod_pyc_count(source_dir)


# The bytecode lands beside the source when nothing redirects it, and under the
# prefix -- with nothing beside the source at all -- when the option does.  The
# tag in the filename is the runtime's own, so only the count is compared.
assert where_the_pyc_lands(False) == (0, 1)
assert where_the_pyc_lands(True) == (1, 0)
print("OK")
