# pyre-check: platforms=linux
# pyre-check: pypy-diverges: pypy3 prints `SAMENAME False` -- the startup file's
#   `co_filename` is not the `__file__` it bound (measured on the ubuntu runner).
#
# APFS and NTFS reject a filename byte with no UTF-8 spelling, so this file
# cannot be created on darwin or win32 at all; ext4 takes every byte but `/`
# and NUL.
#
# CPython-suite gap: `test_cmd_line.test_run_startup` runs a startup file under
# an ASCII name and never varies it, and the undecodable-path handling `test_os`
# covers never reaches PYTHONSTARTUP.  That module also sits at IMPORTERROR in
# the baseline, so none of it runs either way.
#
# parity-tests reason: `PYTHONSTARTUP` is read with `var_os`, which makes it the
# one launcher path an undecodable filename reaches -- `parse_args` refuses a
# non-Unicode argv outright.  A lossy conversion is invisible until the script
# opens its own `__file__`, and it splits `__file__` from `co_filename`.  The
# properties are asserted rather than the escaped spelling, so the row does not
# also pin how each runtime renders a lone surrogate.
import os
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    startup = os.path.join(os.fsencode(tmp), b"start\xff.py")
    with open(startup, "wb") as f:
        f.write(
            b"import os\n"
            b"print('SELFOPEN', open(__file__, 'rb').read(6) == b'import')\n"
            b"print('SAMENAME', (lambda: 0).__code__.co_filename == __file__)\n"
            b"print('BYTES', os.fsencode(__file__).endswith(b'start\\xff.py'))\n"
        )

    done = subprocess.run(
        [sys.executable, "-i"],
        input="",
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONSTARTUP=os.fsdecode(startup)),
    )
    out, err, code = done.stdout, done.stderr, done.returncode

    # `__file__` re-encodes to the bytes the file was opened by, so the script
    # can open itself.  A lossy decode substitutes U+FFFD and the open raises.
    assert "SELFOPEN True" in out, (out, err)
    # And `co_filename` is that same spelling: the compile is named by the same
    # bytes the read was, not by a replacement-character sibling.
    assert "SAMENAME True" in out, (out, err)
    assert "BYTES True" in out, (out, err)
    assert code == 0, (code, err)

print("OK")
