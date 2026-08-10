"""`-W`, `-X` and PYTHONWARNINGS carry a value with no UTF-8 spelling.

These are free text, not identifiers: `app_main.py:785-786` splits an `-X`
value on the first `=` and puts both halves into `sys._xoptions` verbatim, and
`:892-906` appends the `-W` values and the PYTHONWARNINGS pieces to
`sys.warnoptions` verbatim. None of them is required to be spellable in UTF-8,
so a byte the filesystem encoding cannot spell arrives as the surrogate escape
that re-encodes to that byte — in the `_xoptions` key as much as in its value.

An option value never reaches the filesystem, so like
`argv_undecodable_argument.py` this needs no such name on disk and passes the
value to a child instead. Windows takes a wide command line and has no byte
argv, so this shape does not exist there.
"""

import os
import subprocess
import sys

if sys.platform == "win32":
    print("OK")
    raise SystemExit

ESC = os.fsdecode(b"\xff")
assert ESC == "\udcff", ascii(ESC)


def child(*args, env=None):
    result = subprocess.run(
        [sys.executable, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    return result.stdout.decode()


# -W keeps the value it was given. The warnings module rejects it as a filter
# later — it is not a valid action — but that is a separate stage, and the
# option list records what the command line said.
out = child("-W", "ignore" + ESC, "-c", "import sys; print(ascii(sys.warnoptions))")
assert "'ignore\\udcff'" in out, out

# -X splits on the first `=`; the value half keeps the escape.
out = child("-X", "k=v" + ESC, "-c", "import sys; print(ascii(sys._xoptions))")
assert out.strip() == "{'k': 'v\\udcff'}", out

# ... and so does the key half, which is a dict key, not a name.
out = child("-X", "k" + ESC + "=v", "-c", "import sys; print(ascii(sys._xoptions))")
assert out.strip() == "{'k\\udcff': 'v'}", out

# A bare -X with no `=` is the key, and its value is True.
out = child("-X", "bare" + ESC, "-c", "import sys; print(ascii(sys._xoptions))")
assert out.strip() == "{'bare\\udcff': True}", out

# Only the first `=` splits, so a value may carry more of them.
out = child("-X", "k=a=b" + ESC, "-c", "import sys; print(ascii(sys._xoptions))")
assert out.strip() == "{'k': 'a=b\\udcff'}", out

# PYTHONWARNINGS is the same free text arriving through the environment, and it
# is comma-separated: one undecodable piece must not cost the whole variable.
env = dict(os.environ)
env["PYTHONWARNINGS"] = "ignore" + ESC + ",error"
out = child("-c", "import sys; print(ascii(sys.warnoptions))", env=env)
assert "'ignore\\udcff'" in out, out
assert "'error'" in out, out

print("OK")
# CPython-suite gap: command-line tests omit this undecodable option value.
# It is a generic launcher contract, so it belongs in snippets.
