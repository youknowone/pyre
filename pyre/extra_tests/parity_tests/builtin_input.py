import io
import sys


saved_stdin = sys.stdin
saved_stdout = sys.stdout
saved_stderr = sys.stderr
try:
    sys.stdin = io.StringIO("first line\nsecond line")
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    assert input() == "first line"
    assert input("prompt: ") == "second line"
    assert sys.stdout.getvalue() == "prompt: "

    sys.stdin = io.StringIO("")
    try:
        input()
    except EOFError:
        pass
    else:
        raise AssertionError("empty stdin did not raise EOFError")

    sys.stdin = io.StringIO("    whitespace\n")
    assert input() == "    whitespace"
finally:
    sys.stdin = saved_stdin
    sys.stdout = saved_stdout
    sys.stderr = saved_stderr

print("OK")
