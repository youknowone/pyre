"""An unpaired surrogate in a traceback reaches stderr as a backslash escape.

`sys.stderr` is opened with `errors='backslashreplace'`, so an attribute name
carrying a lone surrogate prints as the six-character `\\udcff` escape.  Writing
the three raw WTF-8 bytes behind that code point instead is not the same output
and is not valid UTF-8, so the difference is visible to any consumer that
decodes the stream.

The exception value itself keeps the surrogate: `str(e)` holds the real code
point, and only the encode on the way out spends the escape.
"""

import subprocess
import sys

SCRIPT = r'''
S = "z\udcffz"


class C:
    pass


getattr(C(), S)
'''

EXPECTED = rb"AttributeError: 'C' object has no attribute 'z\udcffz'"


def check_uncaught_render():
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
    )
    assert proc.returncode == 1, proc.returncode
    lines = proc.stderr.splitlines()
    assert lines, proc.stderr
    assert lines[-1] == EXPECTED, lines[-1]
    # The escape is the only thing on that line that is not plain ASCII: a raw
    # surrogate would have left three bytes above 0x7f behind.
    assert lines[-1].isascii(), lines[-1]


def check_group_render():
    script = (
        'raise ExceptionGroup("g\\udcffg", [ValueError("v\\udcffv")])\n'
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert proc.returncode == 1, proc.returncode
    err = proc.stderr
    assert err.isascii(), err
    assert rb"g\udcffg" in err, err
    assert rb"v\udcffv" in err, err


def check_group_str_keeps_the_code_point():
    # `BaseExceptionGroup.__str__` interpolates `self.message`, and a message
    # carrying a lone surrogate interpolates as itself.  Assembling the result
    # through a lossy UTF-8 encode instead substitutes U+FFFD, which makes
    # `str(group)` disagree with `group.message` -- a loss visible here, not
    # only on the way to stderr.
    message = "g\udcffg"
    group = ExceptionGroup(message, [ValueError("v")])
    assert group.message == message, ascii(group.message)
    assert str(group) == message + " (1 sub-exception)", ascii(str(group))
    assert "�" not in str(group), ascii(str(group))


def check_value_keeps_the_code_point():
    S = "z\udcffz"

    class C:
        pass

    try:
        getattr(C(), S)
    except AttributeError as e:
        assert e.name == S, ascii(e.name)
        # The real code point, not the six characters that spell its escape.
        assert str(e) == "'C' object has no attribute '%s'" % S, ascii(str(e))
        assert "\\udcff" not in str(e), ascii(str(e))
    else:
        raise AssertionError("no AttributeError")


check_uncaught_render()
check_group_render()
check_group_str_keeps_the_code_point()
check_value_keeps_the_code_point()
print("OK")
